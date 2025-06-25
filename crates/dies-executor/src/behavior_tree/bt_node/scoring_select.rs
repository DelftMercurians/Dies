use dies_core::debug_tree_node;
use rhai::Engine;

use super::{
    super::bt_callback::BtCallback,
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct ScoringSelectNode {
    children_with_scorers: Vec<(BehaviorNode, BtCallback<f64>)>,
    hysteresis_margin: f64,
    description_text: String,
    node_id_fragment: String,
    current_best_child_index: Option<usize>,
    current_best_child_score: f64,
}

impl ScoringSelectNode {
    pub fn new(
        children_with_scorers: Vec<(BehaviorNode, BtCallback<f64>)>,
        hysteresis_margin: f64,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| "ScoringSelect".to_string());
        Self {
            children_with_scorers,
            hysteresis_margin,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_best_child_index: None,
            current_best_child_score: f64::NEG_INFINITY,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation, engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false, // We don't know if it's active without ticking
            "ScoringSelect",
            Some(format!("{} options", self.children_with_scorers.len())),
            Some(format!("Hysteresis: {:.2}", self.hysteresis_margin)),
        );

        // Debug all child nodes
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        for (child, _) in &self.children_with_scorers {
            child.debug_all_nodes(&child_situation, engine);
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        if self.children_with_scorers.is_empty() {
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

        let mut new_highest_score = f64::NEG_INFINITY;
        let mut new_best_child_candidate_index = 0;
        let mut scores = Vec::with_capacity(self.children_with_scorers.len());

        for (i, (_, scorer)) in self.children_with_scorers.iter().enumerate() {
            let score = scorer.call(situation, engine).unwrap_or(f64::NEG_INFINITY);
            scores.push(score);
            if score > new_highest_score {
                new_highest_score = score;
                new_best_child_candidate_index = i;
            }
        }

        let final_child_to_tick_idx: usize;
        if let Some(active_child_idx) = self.current_best_child_index {
            let score_of_active_child = scores[active_child_idx];
            if score_of_active_child >= new_highest_score - self.hysteresis_margin {
                final_child_to_tick_idx = active_child_idx;
            } else {
                final_child_to_tick_idx = new_best_child_candidate_index;
            }
        } else {
            final_child_to_tick_idx = new_best_child_candidate_index;
        }

        if self.current_best_child_index != Some(final_child_to_tick_idx) {
            self.current_best_child_index = Some(final_child_to_tick_idx);
            self.current_best_child_score = scores[final_child_to_tick_idx];
        }

        let (child_node, _) = &mut self.children_with_scorers[final_child_to_tick_idx];
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        let (status, input) = child_node.tick(&mut child_situation, engine);

        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.current_best_child_index = None;
                self.current_best_child_score = f64::NEG_INFINITY;
            }
            BehaviorStatus::Running => {}
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let selected_info = format!(
            "Selected: {} (Score: {:.2})",
            final_child_to_tick_idx, scores[final_child_to_tick_idx]
        );
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        self.children_with_scorers
            .iter()
            .map(|(c, _)| c.get_full_node_id(&self_full_id))
            .collect()
    }
}
