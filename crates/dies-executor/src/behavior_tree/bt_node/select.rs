use dies_core::debug_tree_node;
use rhai::Engine;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct SelectNode {
    children: Vec<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl SelectNode {
    pub fn new(children: Vec<BehaviorNode>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Select".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let mut final_status = BehaviorStatus::Failure;
        let mut final_input: Option<PlayerControlInput> = None;

        for child in self.children.iter_mut() {
            match child.tick(situation, engine) {
                (BehaviorStatus::Success, input_opt) => {
                    final_status = BehaviorStatus::Success;
                    final_input = input_opt;
                    break;
                }
                (BehaviorStatus::Running, input_opt) => {
                    final_status = BehaviorStatus::Running;
                    final_input = input_opt;
                    break;
                }
                (BehaviorStatus::Failure, _) => continue,
            }
        }

        let is_active =
            final_status == BehaviorStatus::Running || final_status == BehaviorStatus::Success;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            is_active,
        );
        (final_status, final_input)
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
        self.children
            .iter()
            .map(|c| c.get_full_node_id(&self_full_id))
            .collect()
    }
}
