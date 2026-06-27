use std::sync::Arc;

use dies_core::{debug_tree_node, TeamColor};

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::{behavior_tree::BtCallback, control::PlayerControlInput};

pub struct CommittingGuardNode {
    entry_condition: Arc<dyn BtCallback<bool>>,
    cancel_condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
    is_committed: bool,
}

impl CommittingGuardNode {
    pub fn new(
        entry_condition: Arc<dyn BtCallback<bool>>,
        cancel_condition: Arc<dyn BtCallback<bool>>,
        child: BehaviorNode,
        description: String,
    ) -> Self {
        Self {
            entry_condition,
            cancel_condition,
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&description),
            description_text: description,
            is_committed: false,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let entry_result = (self.entry_condition)(situation);
        let cancel_result = (self.cancel_condition)(situation);

        let state_info = if self.is_committed {
            format!(
                "COMMITTED (Cancel: {})",
                if cancel_result { "✓" } else { "✗" }
            )
        } else {
            format!("WAITING (Entry: {})", if entry_result { "✓" } else { "✗" })
        };

        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false, // We don't know if it's active without ticking
            "CommittingGuard",
            Some(state_info),
            Some(self.description()),
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

        // Check if we should commit (start the child)
        if !self.is_committed {
            if (self.entry_condition)(situation) {
                self.is_committed = true;
            } else {
                // Entry condition not met, return failure
                let state_info = format!("WAITING (Entry: ✗)");
                debug_tree_node(
                    situation.debug_key(node_full_id.clone()),
                    self.description(),
                    node_full_id.clone(),
                    self.get_child_node_ids(&situation.viz_path_prefix),
                    false,
                    "CommittingGuard",
                    Some(state_info),
                    Some(self.description()),
                );
                return (BehaviorStatus::Failure, None);
            }
        }

        // We are committed, check if we should cancel
        let cancel_result = (self.cancel_condition)(situation);
        if cancel_result {
            self.is_committed = false;
            let state_info = format!("CANCELLED");
            debug_tree_node(
                situation.debug_key(node_full_id.clone()),
                self.description(),
                node_full_id.clone(),
                self.get_child_node_ids(&situation.viz_path_prefix),
                false,
                "CommittingGuard",
                Some(state_info),
                Some(self.description()),
            );
            return (BehaviorStatus::Failure, None);
        }

        // Run the child
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        let (status, input) = self.child.tick(&mut child_situation);

        // If child completes (success or failure), reset commitment
        match status {
            BehaviorStatus::Success | BehaviorStatus::Failure => {
                self.is_committed = false;
            }
            BehaviorStatus::Running => {
                // Continue commitment
            }
        }

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let state_info = format!(
            "COMMITTED (Cancel: {})",
            if cancel_result { "✓" } else { "✗" }
        );

        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active,
            "CommittingGuard",
            Some(state_info),
            Some(self.description()),
        );

        (status, input)
    }

    pub fn description(&self) -> String {
        format!(
            "WHEN ({}) COMMIT TO ({}) UNTIL ({})",
            self.description_text,
            self.child.description(),
            "cancel condition"
        )
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

impl From<CommittingGuardNode> for BehaviorNode {
    fn from(node: CommittingGuardNode) -> Self {
        BehaviorNode::CommittingGuard(node)
    }
}

pub struct CommittingGuardNodeBuilder {
    entry_condition: Option<Arc<dyn BtCallback<bool>>>,
    cancel_condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
    description: Option<String>,
}

impl CommittingGuardNodeBuilder {
    pub fn new() -> Self {
        Self {
            entry_condition: None,
            cancel_condition: None,
            child: None,
            description: None,
        }
    }

    pub fn when(mut self, condition: impl BtCallback<bool>) -> Self {
        self.entry_condition = Some(Arc::new(condition));
        self
    }

    pub fn commit_to(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn until(mut self, condition: impl BtCallback<bool>) -> Self {
        self.cancel_condition = Some(Arc::new(condition));
        self
    }

    pub fn description(mut self, description: impl AsRef<str>) -> Self {
        self.description = Some(description.as_ref().to_string());
        self
    }

    pub fn build(self) -> CommittingGuardNode {
        CommittingGuardNode::new(
            self.entry_condition.expect("Entry condition is required"),
            self.cancel_condition.expect("Cancel condition is required"),
            self.child.expect("Child is required"),
            self.description.unwrap_or_default(),
        )
    }
}

pub fn committing_guard_node() -> CommittingGuardNodeBuilder {
    CommittingGuardNodeBuilder::new()
}
