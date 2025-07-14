use dies_core::{debug_tree_node, PlayerId};

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct SemaphoreNode {
    child: Box<BehaviorNode>,
    semaphore_id_str: String,
    max_count: usize,
    description_text: String,
    node_id_fragment: String,
    player_id_holding_via_this_node: Option<PlayerId>,
}

impl SemaphoreNode {
    pub fn new(
        child: BehaviorNode,
        semaphore_id: String,
        max_count: usize,
        description: Option<String>,
    ) -> Self {
        let desc = description.unwrap_or_else(|| format!("Semaphore_{}", semaphore_id));
        Self {
            child: Box::new(child),
            semaphore_id_str: semaphore_id,
            max_count,
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
            player_id_holding_via_this_node: None,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false, // We don't know if it's active without ticking
            "Semaphore",
            Some(format!("Max: {}", self.max_count)),
            Some(format!("ID: {}", self.semaphore_id_str)),
        );

        // Debug the child node
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        self.child.debug_all_nodes(&child_situation);
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let player_id = situation.player_id;
        let mut result_status = BehaviorStatus::Failure;
        let mut result_input: Option<PlayerControlInput> = None;

        // Check if we have a local claim but verify it's still valid in the global context
        let acquired_semaphore = if self.player_id_holding_via_this_node == Some(player_id) {
            // Verify our local state matches global state
            if situation.bt_context.try_acquire_semaphore(
                &self.semaphore_id_str,
                self.max_count,
                player_id,
            ) {
                true
            } else {
                // Local state is out of sync, clear it
                self.player_id_holding_via_this_node = None;
                false
            }
        } else {
            if situation.bt_context.try_acquire_semaphore(
                &self.semaphore_id_str,
                self.max_count,
                player_id,
            ) {
                self.player_id_holding_via_this_node = Some(player_id);
                true
            } else {
                false
            }
        };

        if acquired_semaphore {
            let mut child_situation = situation.clone();
            child_situation.viz_path_prefix = node_full_id.clone();
            let (child_status, child_input) = self.child.tick(&mut child_situation);
            result_status = child_status;
            result_input = child_input;

            match child_status {
                BehaviorStatus::Success | BehaviorStatus::Failure => {
                    situation
                        .bt_context
                        .release_semaphore(&self.semaphore_id_str, player_id);
                    self.player_id_holding_via_this_node = None;
                }
                BehaviorStatus::Running => {}
            }
        } else {
            // Ensure local state is cleared when we can't acquire
            self.player_id_holding_via_this_node = None;
            result_status = BehaviorStatus::Failure;
            result_input = None;
        }

        let is_active = result_status == BehaviorStatus::Running
            || (result_status == BehaviorStatus::Success && acquired_semaphore);
        let semaphore_state = if acquired_semaphore {
            "Acquired"
        } else {
            "Waiting"
        };
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
            "Semaphore",
            Some(format!("{} (Max: {})", semaphore_state, self.max_count)),
            Some(format!("ID: {}", self.semaphore_id_str)),
        );
        (result_status, result_input)
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
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}

pub struct SemaphoreNodeBuilder {
    child: Option<BehaviorNode>,
    semaphore_id: Option<String>,
    max_count: Option<usize>,
    description: Option<String>,
}

impl SemaphoreNodeBuilder {
    pub fn new() -> Self {
        Self {
            child: None,
            semaphore_id: None,
            max_count: None,
            description: None,
        }
    }

    pub fn do_then(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn semaphore_id(mut self, semaphore_id: String) -> Self {
        self.semaphore_id = Some(semaphore_id);
        self
    }

    pub fn max_entry(mut self, max_count: usize) -> Self {
        self.max_count = Some(max_count);
        self
    }

    pub fn description(mut self, description: impl AsRef<str>) -> Self {
        self.description = Some(description.as_ref().to_string());
        self
    }

    pub fn build(self) -> SemaphoreNode {
        SemaphoreNode::new(
            self.child.expect("Child is required"),
            self.semaphore_id.expect("Semaphore ID is required"),
            self.max_count.expect("Max count is required"),
            self.description,
        )
    }
}

pub fn semaphore_node() -> SemaphoreNodeBuilder {
    SemaphoreNodeBuilder::new()
}

impl From<SemaphoreNode> for BehaviorNode {
    fn from(node: SemaphoreNode) -> Self {
        BehaviorNode::Semaphore(node)
    }
}
