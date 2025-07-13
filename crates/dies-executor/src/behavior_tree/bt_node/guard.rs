use std::sync::Arc;

use dies_core::debug_tree_node;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::{behavior_tree::BtCallback, control::PlayerControlInput};

#[derive(Clone)]
pub struct GuardNode {
    condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl GuardNode {
    pub fn new(
        condition: Arc<dyn BtCallback<bool>>,
        child: BehaviorNode,
        description: String,
    ) -> Self {
        Self {
            condition,
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&description),
            description_text: description,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let condition_result = (self.condition)(situation);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false, // We don't know if it's active without ticking
            "Guard",
            Some(format!(
                "Condition: {}",
                if condition_result { "✓" } else { "✗" }
            )),
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
        let (status, input) = if (self.condition)(situation) {
            let mut child_situation = situation.clone();
            child_situation.viz_path_prefix = node_full_id.clone();
            self.child.tick(&mut child_situation)
        } else {
            (BehaviorStatus::Failure, None)
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let condition_state = (self.condition)(situation);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active && condition_state,
            "Guard",
            Some(format!(
                "Condition: {}",
                if condition_state { "✓" } else { "✗" }
            )),
            Some(self.description()),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        format!(
            "IF ({}) THEN ({})",
            self.description_text,
            self.child.description()
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
            format!("{}/{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}

impl From<GuardNode> for BehaviorNode {
    fn from(node: GuardNode) -> Self {
        BehaviorNode::Guard(node)
    }
}

pub struct GuardNodeBuilder {
    condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
    description: Option<String>,
}

impl GuardNodeBuilder {
    pub fn new() -> Self {
        Self {
            condition: None,
            child: None,
            description: None,
        }
    }

    pub fn condition(mut self, condition: impl BtCallback<bool>) -> Self {
        self.condition = Some(Arc::new(condition));
        self
    }

    pub fn then(mut self, child: impl Into<BehaviorNode>) -> Self {
        self.child = Some(child.into());
        self
    }

    pub fn description(mut self, description: impl AsRef<str>) -> Self {
        self.description = Some(description.as_ref().to_string());
        self
    }

    pub fn build(self) -> GuardNode {
        GuardNode::new(
            self.condition.expect("Condition is required"),
            self.child.expect("Child is required"),
            self.description.expect("Description is required"),
        )
    }
}

pub fn guard_node() -> GuardNodeBuilder {
    GuardNodeBuilder::new()
}
