use std::sync::Arc;

use dies_core::debug_tree_node;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    sanitize_id_fragment, BehaviorNode,
};
use crate::{behavior_tree::BtCallback, control::PlayerControlInput};

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
            situation.debug_key(node_full_id.clone()),
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
            situation.debug_key(node_full_id.clone()),
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
            format!("{}.{}", current_path_prefix, fragment)
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
            self.description.unwrap_or_default(),
        )
    }
}

pub fn guard_node() -> GuardNodeBuilder {
    GuardNodeBuilder::new()
}

pub struct GuardWithHysteresisNode {
    condition: Arc<dyn BtCallback<bool>>,
    child: Box<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
    hysteresis_margin: f64,
    last_condition_result: Option<bool>,
    activation_count: u32,
    deactivation_count: u32,
}

impl GuardWithHysteresisNode {
    pub fn new(
        condition: Arc<dyn BtCallback<bool>>,
        child: BehaviorNode,
        description: String,
        hysteresis_margin: f64,
    ) -> Self {
        Self {
            condition,
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&description),
            description_text: description,
            hysteresis_margin,
            last_condition_result: None,
            activation_count: 0,
            deactivation_count: 0,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let condition_result = self.evaluate_condition_with_hysteresis(situation);
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            false,
            "GuardHysteresis",
            Some(format!(
                "Condition: {} (margin: {:.2})",
                if condition_result { "✓" } else { "✗" },
                self.hysteresis_margin
            )),
            Some(self.description()),
        );

        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        self.child.debug_all_nodes(&child_situation);
    }

    fn evaluate_condition_with_hysteresis(&self, situation: &RobotSituation) -> bool {
        let raw_condition = (self.condition)(situation);

        match self.last_condition_result {
            None => raw_condition,
            Some(last_result) => {
                if last_result && !raw_condition {
                    // Was true, now false - require some stability before switching
                    false
                } else if !last_result && raw_condition {
                    // Was false, now true - require some stability before switching
                    true
                } else {
                    // Same as last time, keep it
                    last_result
                }
            }
        }
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let raw_condition = (self.condition)(situation);

        // Apply hysteresis logic
        let effective_condition = match self.last_condition_result {
            None => {
                self.last_condition_result = Some(raw_condition);
                raw_condition
            }
            Some(last_result) => {
                if last_result && !raw_condition {
                    // Was active, now inactive - count deactivation attempts
                    self.deactivation_count += 1;
                    if self.deactivation_count >= (self.hysteresis_margin * 10.0) as u32 {
                        self.last_condition_result = Some(false);
                        self.deactivation_count = 0;
                        self.activation_count = 0;
                        false
                    } else {
                        true // Keep previous state
                    }
                } else if !last_result && raw_condition {
                    // Was inactive, now active - count activation attempts
                    self.activation_count += 1;
                    if self.activation_count >= (self.hysteresis_margin * 10.0) as u32 {
                        self.last_condition_result = Some(true);
                        self.activation_count = 0;
                        self.deactivation_count = 0;
                        true
                    } else {
                        false // Keep previous state
                    }
                } else {
                    // Same as before, reset counters
                    self.activation_count = 0;
                    self.deactivation_count = 0;
                    last_result
                }
            }
        };

        let (status, input) = if effective_condition {
            let mut child_situation = situation.clone();
            child_situation.viz_path_prefix = node_full_id.clone();
            self.child.tick(&mut child_situation)
        } else {
            (BehaviorStatus::Failure, None)
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        debug_tree_node(
            situation.debug_key(node_full_id.clone()),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&situation.viz_path_prefix),
            is_active && effective_condition,
            "GuardHysteresis",
            Some(format!(
                "Condition: {} (raw: {}, act: {}, deact: {})",
                if effective_condition { "✓" } else { "✗" },
                if raw_condition { "✓" } else { "✗" },
                self.activation_count,
                self.deactivation_count
            )),
            Some(self.description()),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        format!(
            "IF ({}) THEN ({}) [hysteresis: {:.2}]",
            self.description_text,
            self.child.description(),
            self.hysteresis_margin
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

impl From<GuardWithHysteresisNode> for BehaviorNode {
    fn from(node: GuardWithHysteresisNode) -> Self {
        BehaviorNode::GuardWithHysteresis(node)
    }
}

pub struct GuardWithHysteresisNodeBuilder {
    condition: Option<Arc<dyn BtCallback<bool>>>,
    child: Option<BehaviorNode>,
    description: Option<String>,
    hysteresis_margin: f64,
}

impl GuardWithHysteresisNodeBuilder {
    pub fn new() -> Self {
        Self {
            condition: None,
            child: None,
            description: None,
            hysteresis_margin: 0.1, // Default 10 ticks
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

    pub fn hysteresis_margin(mut self, margin: f64) -> Self {
        self.hysteresis_margin = margin;
        self
    }

    pub fn build(self) -> GuardWithHysteresisNode {
        GuardWithHysteresisNode::new(
            self.condition.expect("Condition is required"),
            self.child.expect("Child is required"),
            self.description.unwrap_or_default(),
            self.hysteresis_margin,
        )
    }
}

pub fn guard_with_hysteresis_node() -> GuardWithHysteresisNodeBuilder {
    GuardWithHysteresisNodeBuilder::new()
}
