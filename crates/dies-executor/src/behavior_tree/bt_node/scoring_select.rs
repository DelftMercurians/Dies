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

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);

        // if self.children_with_scorers.is_empty() {
        //     debug_tree_node(
        //         format!("bt.p{}.{}", situation.player_id, node_full_id),
        //         self.description(),
        //         node_full_id.clone(),
        //         vec![],
        //         false,
        //     );
        //     return (BehaviorStatus::Failure, None);
        // }

        // let mut new_highest_score = f64::NEG_INFINITY;
        // let mut new_best_child_candidate_index = 0;
        // let mut scores = Vec::with_capacity(self.children_with_scorers.len());

        // for (i, (_, scorer)) in self.children_with_scorers.iter().enumerate() {
        //     let score = scorer(situation);
        //     scores.push(score);
        //     if score > new_highest_score {
        //         new_highest_score = score;
        //         new_best_child_candidate_index = i;
        //     }
        // }

        // let final_child_to_tick_idx: usize;
        // if let Some(active_child_idx) = self.current_best_child_index {
        //     let score_of_active_child = scores[active_child_idx];
        //     if score_of_active_child >= new_highest_score - self.hysteresis_margin {
        //         final_child_to_tick_idx = active_child_idx;
        //     } else {
        //         final_child_to_tick_idx = new_best_child_candidate_index;
        //     }
        // } else {
        //     final_child_to_tick_idx = new_best_child_candidate_index;
        // }

        // if self.current_best_child_index != Some(final_child_to_tick_idx) {
        //     self.current_best_child_index = Some(final_child_to_tick_idx);
        //     self.current_best_child_score = scores[final_child_to_tick_idx];
        // }

        // let (child_node, _) = &mut self.children_with_scorers[final_child_to_tick_idx];
        // let (status, input) = child_node.tick(situation);

        // match status {
        //     BehaviorStatus::Success | BehaviorStatus::Failure => {
        //         self.current_best_child_index = None;
        //         self.current_best_child_score = f64::NEG_INFINITY;
        //     }
        //     BehaviorStatus::Running => {}
        // }

        // let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        // debug_tree_node(
        //     format!("bt.p{}.{}", situation.player_id, node_full_id),
        //     self.description(),
        //     node_full_id.clone(),
        //     self.get_child_node_ids(&node_full_id),
        //     is_active,
        // );
        // (status, input)
        Default::default()
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
        Default::default()
        // self.children_with_scorers
        //     .iter()
        //     .map(|(c, _)| c.get_full_node_id(&self_full_id))
        //     .collect()
    }
}
