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

    pub fn debug_all_nodes(&self, situation: &RobotSituation, engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false, // We don't know if it's active without ticking
            "Select",
            Some(format!("{} children", self.children.len())),
            Some("First success wins".to_string()),
        );

        // Debug all child nodes
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        for child in &self.children {
            child.debug_all_nodes(&child_situation, engine);
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

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();

        for child in self.children.iter_mut() {
            match child.tick(&mut child_situation, engine) {
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
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
            "Select",
            Some(format!("{} children", self.children.len())),
            Some("First success wins".to_string()),
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
