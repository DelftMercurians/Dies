use dies_core::debug_tree_node;
use rhai::{Array, Engine};

use super::{
    super::bt_callback::BtCallback,
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Scorer {
    pub node: BehaviorNode,
    pub callback: BtCallback<f64>,
    pub key: String,
}

#[derive(Clone)]
pub enum ScorersSource {
    Static(Vec<Scorer>),
    DynamicWithKeys(BtCallback<Array>),
}

#[derive(Clone)]
pub struct ScoringSelectNode {
    scorers_source: ScorersSource,
    hysteresis_margin: f64,
    description_text: String,
    node_id_fragment: String,
    current_best_child_key: Option<String>,
    current_best_child_score: f64,
    last_children: Vec<Scorer>,
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers: Vec<(BehaviorNode, BtCallback<f64>)>,
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
            last_children: Vec::new(),
        }
    }

    pub fn new_dynamic(
        callback: BtCallback<Array>,
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
            last_children: Vec::new(),
        }
    }

    fn get_scorers(&self, situation: &RobotSituation, engine: &Engine) -> Vec<Scorer> {
        match &self.scorers_source {
            ScorersSource::Static(scorers) => scorers.clone(),
            ScorersSource::DynamicWithKeys(callback) => callback
                .call(situation, engine)
                .unwrap_or_else(|e| {
                    log::error!("Failed to generate dynamic scorers: {}", e);
                    vec![]
                })
                .into_iter()
                .filter_map(|n| n.try_cast_result::<Scorer>().ok())
                .collect(),
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation, engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let children = self.last_children.clone();
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
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
        for scorer in &children {
            scorer.node.debug_all_nodes(&child_situation, engine);
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let mut children = self.get_scorers(situation, engine);

        if children.is_empty() {
            debug_tree_node(
                format!("bt.p{}.{}", situation.player_id, node_full_id),
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
        for scorer in &children {
            let score = scorer
                .callback
                .call(situation, engine)
                .unwrap_or(f64::NEG_INFINITY);
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
        let (status, input) = child_node.tick(&mut child_situation, engine);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.current_best_child_key = None;
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {}
        }

        self.last_children = children;
        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let selected_info = format!(
            "Selected: {} (Score: {:.2})",
            final_child_key, scores[&final_child_key]
        );
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids_with_children(&situation.viz_path_prefix, &self.last_children),
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
            format!("{}/{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        self.get_child_node_ids_with_children(current_path_prefix, &self.last_children)
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
