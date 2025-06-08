use dies_core::debug_tree_node;
use rhai::Engine;

use super::{
    super::bt_core::{BehaviorStatus, RobotSituation},
    super::situation::Situation,
    sanitize_id_fragment, BehaviorNode,
};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct GuardNode {
    condition: Situation,
    child: Box<BehaviorNode>,
    description_text: String,
    node_id_fragment: String,
}

impl GuardNode {
    pub fn new(
        condition: Situation,
        child: BehaviorNode,
        description_override: Option<String>,
    ) -> Self {
        let desc = description_override.unwrap_or_else(|| {
            format!(
                "Guard_If_{}_Then_{}",
                sanitize_id_fragment(condition.description()),
                child.get_node_id_fragment()
            )
        });
        Self {
            condition,
            child: Box::new(child),
            node_id_fragment: sanitize_id_fragment(&desc),
            description_text: desc,
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation, engine: &Engine) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let condition_result = self.condition.check(situation, engine);
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
            Some(self.condition.description().to_string()),
        );

        // Debug the child node
        let mut child_situation = situation.clone();
        child_situation.viz_path_prefix = node_full_id.clone();
        self.child.debug_all_nodes(&child_situation, engine);
    }

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let (status, input) = if self.condition.check(situation, engine) {
            let mut child_situation = situation.clone();
            child_situation.viz_path_prefix = node_full_id.clone();
            self.child.tick(&mut child_situation, engine)
        } else {
            (BehaviorStatus::Failure, None)
        };

        let is_active = status == BehaviorStatus::Running || status == BehaviorStatus::Success;
        let condition_state = self.condition.check(situation, engine);
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
            Some(self.condition.description().to_string()),
        );
        (status, input)
    }

    pub fn description(&self) -> String {
        format!(
            "IF ({}) THEN ({})",
            self.condition.description(),
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
