use dies_core::debug_tree_node;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::{behavior_tree::BtCallback, control::PlayerControlInput};
use std::{collections::HashMap, sync::Arc};

pub struct Scorer {
    pub node: BehaviorNode,
    pub callback: Arc<dyn BtCallback<f64>>,
    pub key: String,
}

pub enum ScorersSource {
    Static(Vec<Scorer>),
    DynamicWithKeys(Arc<dyn BtCallback<Vec<Scorer>>>),
}

pub struct ScoringSelectNode {
    scorers_source: ScorersSource,
    hysteresis_margin: f64,
    description_text: String,
    node_id_fragment: String,
    current_best_child_key: Option<String>,
    current_best_child_score: f64,
    last_children_ids: Vec<String>,
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers: Vec<(BehaviorNode, Arc<dyn BtCallback<f64>>)>,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "ScoringSelect".to_string());
        let scorers = children_with_scorers
            .into_iter()
            .enumerate()
            .map(|(i, (node, callback))| Scorer {
                node,
                callback,
                key: i.to_string(), // Use index as key for static lists
            })
            .collect();
        Self {
            scorers_source: ScorersSource::Static(scorers),
            hysteresis_margin,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_best_child_key: None,
            current_best_child_score: f64::NEG_INFINITY,
            last_children_ids: Vec::new(),
        }
    }

    pub fn new_dynamic(
        callback: Arc<dyn BtCallback<Vec<Scorer>>>,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "ScoringSelect".to_string());
        Self {
            scorers_source: ScorersSource::DynamicWithKeys(callback),
            hysteresis_margin,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_best_child_key: None,
            current_best_child_score: f64::NEG_INFINITY,
            last_children_ids: Vec::new(),
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        let children = match &self.scorers_source {
            ScorersSource::Static(scorers) => scorers,
            ScorersSource::DynamicWithKeys(_) => panic!("Dynamic children not supported"),
        };
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids_with_children(&situation.viz_path_prefix, &children),
            false, // We don't know if it's active without ticking
            "ScoringSelect",
            Some(format!("{} options", children.len())),
            Some(format!("Hysteresis: {:.2}", self.hysteresis_margin)),
        );

        // Debug all child nodes
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        match &self.scorers_source {
            ScorersSource::Static(scorers) => {
                for scorer in scorers {
                    scorer.node.debug_all_nodes(&child_situation);
                }
            }
            _ => {}
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        self.last_children_ids = match &self.scorers_source {
            ScorersSource::Static(scorers) => {
                self.get_child_node_ids_with_children(&situation.viz_path_prefix, scorers)
            }
            ScorersSource::DynamicWithKeys(cb) => {
                let children = cb(situation);

                self.get_child_node_ids_with_children(&situation.viz_path_prefix, &children)
            }
        };

        if self.last_children_ids.is_empty() {
            debug_tree_node(
                situation.debug_key(node_full_id.clone()),
                self.description(),
                node_full_id.clone(),
                vec![],
                false,
                "ScoringSelect",
                Some("Empty".to_string()),
                Some("No children".to_string()),
            );
            return (BehaviorStatus::Failure, None);
        }

        let mut scores = HashMap::new();
        let children = match &mut self.scorers_source {
            ScorersSource::Static(scorers) => scorers,
            ScorersSource::DynamicWithKeys(cb) => panic!("Dynamic children not supported"),
        };
        for scorer in children.iter() {
            let score = (scorer.callback)(situation);
            scores.insert(scorer.key.clone(), score);
        }

        let (best_key, &highest_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let final_child_key = if let Some(active_child_key) = self.current_best_child_key.as_ref() {
            if let Some(&score_of_active_child) = scores.get(active_child_key) {
                if score_of_active_child >= highest_score - self.hysteresis_margin {
                    active_child_key.clone()
                } else {
                    best_key.clone()
                }
            } else {
                best_key.clone()
            }
        } else {
            best_key.clone()
        };

        if self.current_best_child_key.as_ref() != Some(&final_child_key) {
            self.current_best_child_key = Some(final_child_key.clone());
            self.current_best_child_score = scores[&final_child_key];
        }

        let child_to_tick_idx = children
            .iter()
            .position(|c| c.key == final_child_key)
            .unwrap();
        let (child_node, _) = {
            let scorer = &mut children[child_to_tick_idx];
            (&mut scorer.node, &scorer.callback)
        };

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        let (status, input) = child_node.tick(&mut child_situation);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.current_best_child_key = None;
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {}
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let selected_info = format!(
            "Selected: {} (Score: {:.2})",
            final_child_key, scores[&final_child_key]
        );
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.last_children_ids.clone(),
            is_active,
            "ScoringSelect",
            Some(selected_info),
            Some(format!("Hysteresis: {:.2}", self.hysteresis_margin)),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        self.description_text.clone()
    }

    pub fn get_node_id_fragment(&self) -> String {
        self.node_id_fragment.clone()
    }

    pub fn get_full_node_id(&self, current_path_prefix: &str) -> String {
        let fragment = self.get_node_id_fragment();
        if current_path_prefix.is_empty() {
            fragment
        } else {
            format!("{}.{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        self.last_children_ids.clone()
    }

    fn get_child_node_ids_with_children(
        &self,
        current_path_prefix: &str,
        children: &[Scorer],
    ) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        children
            .iter()
            .map(|c| c.node.get_full_node_id(&self_full_id))
            .collect()
    }
}

impl From<ScoringSelectNode> for BehaviorNode {
    fn from(node: ScoringSelectNode) -> Self {
        BehaviorNode::ScoringSelect(node)
    }
}

pub struct ScoringSelectNodeBuilder {
    children_with_scorers: Vec<(BehaviorNode, Arc<dyn BtCallback<f64>>)>,
    hysteresis_margin: f64,
    description: Option<String>,
}

impl ScoringSelectNodeBuilder {
    pub fn new() -> Self {
        Self {
            children_with_scorers: Vec::new(),
            hysteresis_margin: 0.1,
            description: None,
        }
    }

    pub fn dynamic(self, callback: impl BtCallback<Vec<Scorer>>) -> ScoringSelectNode {
        ScoringSelectNode::new_dynamic(Arc::new(callback), self.hysteresis_margin, self.description)
    }

    pub fn add_child(
        mut self,
        child: impl Into<BehaviorNode>,
        scorer: impl BtCallback<f64>,
    ) -> Self {
        self.children_with_scorers
            .push((child.into(), Arc::new(scorer)));
        self
    }

    pub fn hysteresis_margin(mut self, hysteresis_margin: f64) -> Self {
        self.hysteresis_margin = hysteresis_margin;
        self
    }

    pub fn description(mut self, description: impl AsRef<str>) -> Self {
        self.description = Some(description.as_ref().to_string());
        self
    }

    pub fn build(self) -> ScoringSelectNode {
        ScoringSelectNode::new(
            self.children_with_scorers,
            self.hysteresis_margin,
            self.description,
        )
    }
}

pub fn scoring_select_node() -> ScoringSelectNodeBuilder {
    ScoringSelectNodeBuilder::new()
}
