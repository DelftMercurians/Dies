use super::bt_core::{BehaviorStatus, RobotSituation};
use crate::control::PlayerControlInput;
use rhai::Engine;

mod action;
mod guard;
mod noop;
mod scoring_select;
mod select;
mod semaphore;
mod sequence;

pub use action::*;
pub use guard::GuardNode;
pub use noop::NoopNode;
pub use scoring_select::ScoringSelectNode;
pub use select::SelectNode;
pub use semaphore::SemaphoreNode;
pub use sequence::SequenceNode;

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
        engine: &Engine,
    ) -> (BehaviorStatus, Option<PlayerControlInput>) {
        match self {
            BehaviorNode::Select(node) => node.tick(situation, engine),
            BehaviorNode::Sequence(node) => node.tick(situation, engine),
            BehaviorNode::Guard(node) => node.tick(situation, engine),
            BehaviorNode::Action(node) => node.tick(situation, engine),
            BehaviorNode::Semaphore(node) => node.tick(situation, engine),
            BehaviorNode::ScoringSelect(node) => node.tick(situation, engine),
            BehaviorNode::Noop(_) => (BehaviorStatus::Success, None),
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
