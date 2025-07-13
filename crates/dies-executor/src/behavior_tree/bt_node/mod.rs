use super::bt_core::{BehaviorStatus, RobotSituation};
use crate::control::PlayerControlInput;

mod action;
mod guard;
mod noop;
mod scoring_select;
mod select;
mod semaphore;
mod sequence;

pub use action::*;
pub use guard::{guard_node, GuardNode};
pub use noop::{noop_node, NoopNode};
pub use scoring_select::{scoring_select_node, ScoringSelectNode};
pub use select::{select_node, SelectNode};
pub use semaphore::{semaphore_node, SemaphoreNode};
pub use sequence::{sequence_node, SequenceNode};

#[derive(Clone)]
pub enum BehaviorNode {
    Select(SelectNode),
    Sequence(SequenceNode),
    Guard(GuardNode),
    Action(ActionNode),
    Semaphore(SemaphoreNode),
    ScoringSelect(ScoringSelectNode),
    Noop(NoopNode),
}

impl BehaviorNode {
    pub fn tick(
        &mut self,
        situation: &mut RobotSituation,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        match self {
            BehaviorNode::Select(node) => node.tick(situation),
            BehaviorNode::Sequence(node) => node.tick(situation),
            BehaviorNode::Guard(node) => node.tick(situation),
            BehaviorNode::Action(node) => node.tick(situation),
            BehaviorNode::Semaphore(node) => node.tick(situation),
            BehaviorNode::ScoringSelect(node) => node.tick(situation),
            BehaviorNode::Noop(_) => (BehaviorStatus::Success, None),
        }
    }

    pub fn debug_all_nodes(&self, situation: &RobotSituation) {
        match self {
            BehaviorNode::Select(node) => node.debug_all_nodes(situation),
            BehaviorNode::Sequence(node) => node.debug_all_nodes(situation),
            BehaviorNode::Guard(node) => node.debug_all_nodes(situation),
            BehaviorNode::Action(node) => node.debug_all_nodes(situation),
            BehaviorNode::Semaphore(node) => node.debug_all_nodes(situation),
            BehaviorNode::ScoringSelect(node) => node.debug_all_nodes(situation),
            BehaviorNode::Noop(_) => {
                // Noop nodes don't have structure to debug
            }
        }
    }

    pub fn description(&self) -> String {
        match self {
            BehaviorNode::Select(node) => node.description(),
            BehaviorNode::Sequence(node) => node.description(),
            BehaviorNode::Guard(node) => node.description(),
            BehaviorNode::Action(node) => node.description(),
            BehaviorNode::Semaphore(node) => node.description(),
            BehaviorNode::ScoringSelect(node) => node.description(),
            BehaviorNode::Noop(_) => "Noop".to_string(),
        }
    }

    pub fn get_node_id_fragment(&self) -> String {
        match self {
            BehaviorNode::Select(node) => node.get_node_id_fragment(),
            BehaviorNode::Sequence(node) => node.get_node_id_fragment(),
            BehaviorNode::Guard(node) => node.get_node_id_fragment(),
            BehaviorNode::Action(node) => node.get_node_id_fragment(),
            BehaviorNode::Semaphore(node) => node.get_node_id_fragment(),
            BehaviorNode::ScoringSelect(node) => node.get_node_id_fragment(),
            BehaviorNode::Noop(_) => "Noop".to_string(),
        }
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
        match self {
            BehaviorNode::Select(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Sequence(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Guard(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Action(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Semaphore(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::ScoringSelect(node) => node.get_child_node_ids(current_path_prefix),
            BehaviorNode::Noop(_) => vec![],
        }
    }
}

pub(self) fn sanitize_id_fragment(text: &str) -> String {
    text.chars()
        .filter_map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' => Some(c.to_ascii_lowercase()),
            ' ' | '_' | '-' => Some('_'),
            _ => None,
        })
        .collect::<String>()
}
