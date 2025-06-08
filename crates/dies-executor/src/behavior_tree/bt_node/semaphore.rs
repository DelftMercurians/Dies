use dies_core::{debug_tree_node, PlayerId};
use rhai::Engine;

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

    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        let node_full_id = self.get_full_node_id(&situation.viz_path_prefix);
        let player_id = situation.player_id;
        let mut result_status = BehaviorStatus::Failure;
        let mut result_input: Option<PlayerControlInput> = None;

        let acquired_semaphore = if self.player_id_holding_via_this_node == Some(player_id) {
            true
        } else {
            if situation.team_context.try_acquire_semaphore(
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
            let (child_status, child_input) = self.child.tick(situation, engine);
            result_status = child_status;
            result_input = child_input;

            match child_status {
                BehaviorStatus::Success | BehaviorStatus::Failure => {
                    situation
                        .team_context
                        .release_semaphore(&self.semaphore_id_str, player_id);
                    self.player_id_holding_via_this_node = None;
                }
                BehaviorStatus::Running => {}
            }
        } else {
            result_status = BehaviorStatus::Failure;
            result_input = None;
        }

        let is_active = result_status == BehaviorStatus::Running
            || (result_status == BehaviorStatus::Success && acquired_semaphore);
        debug_tree_node(
            format!("bt.p{}.{}", situation.player_id, node_full_id),
            self.description(),
            node_full_id.clone(),
            self.get_child_node_ids(&node_full_id),
            is_active,
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
            format!("{}/{}", current_path_prefix, fragment)
        }
    }

    pub fn get_child_node_ids(&self, current_path_prefix: &str) -> Vec<String> {
        let self_full_id = self.get_full_node_id(current_path_prefix);
        vec![self.child.get_full_node_id(&self_full_id)]
    }
}
