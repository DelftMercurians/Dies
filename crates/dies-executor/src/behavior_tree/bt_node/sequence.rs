use dies_core::debug_tree_node;
use rhai::Engine;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct SequenceNode {
    children: Vec<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
    current_child_index: usize,
}

impl SequenceNode {
    pub fn new(children: Vec<BehaviorNode>, description: Option<String>) -> Self {
        let desc = description.unwrap_or_else(|| "Sequence".to_string());
        Self {
            children,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            current_child_index: 0,
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
            "Sequence",
            Some(format!(
                "{}/{} children",
                self.current_child_index,
                self.children.len()
            )),
            Some("All must succeed in order".to_string()),
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
        let mut last_input_on_success: Option<PlayerControlInput> = None;

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();

        while self.current_child_index < self.children.len() {
            match self.children[self.current_child_index].tick(&mut child_situation, engine) {
                (BehaviorStatus::Success, input_opt) => {
                    self.current_child_index += 1;
                    last_input_on_success = input_opt;
                }
                (BehaviorStatus::Running, input_opt) => {
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&situation.viz_path_prefix),
                        true,
                        "Sequence",
                        Some(format!(
                            "{}/{} children",
                            self.current_child_index + 1,
                            self.children.len()
                        )),
                        Some("All must succeed in order".to_string()),
                    );
                    return (BehaviorStatus::Running, input_opt);
                }
                (BehaviorStatus::Failure, _input_opt) => {
                    self.current_child_index = 0;
                    debug_tree_node(
                        format!("bt.p{}.{}", situation.player_id, node_full_id),
                        self.description(),
                        node_full_id.clone(),
                        self.get_child_node_ids(&situation.viz_path_prefix),
                        false,
                        "Sequence",
                        Some(format!(
                            "Failed at {}/{}",
                            self.current_child_index + 1,
                            self.children.len()
                        )),
                        Some("All must succeed in order".to_string()),
                    );
                    return (BehaviorStatus::Failure, None);
                }
            }
        }

        self.current_child_index = 0;
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            true,
            "Sequence",
            Some(format!(
                "All {}/{} completed",
                self.children.len(),
                self.children.len()
            )),
            Some("All must succeed in order".to_string()),
        );
        (BehaviorStatus::Success, last_input_on_success)
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
