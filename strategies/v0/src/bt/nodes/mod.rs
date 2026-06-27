//! Behavior tree node types. Each node's `tick` returns a `(BehaviorStatus,
//! Option<SkillCommand>)`: the status drives the tree's control flow, and the
//! optional command is the skill the active leaf wants this robot to run this
//! frame (`None` = leave the robot's current skill untouched).

mod action;
mod control;
mod movement;

pub use action::*;
pub use control::*;
pub use movement::*;

use dies_strategy_protocol::SkillCommand;

use super::situation::{BehaviorStatus, RobotSituation};

pub type TickResult = (BehaviorStatus, Option<SkillCommand>);

/// Tagged union over every node kind, so trees can be built as owned, nested
/// values without boxing every node behind a trait object.
pub enum BehaviorNode {
    Select(SelectNode),
    Sequence(SequenceNode),
    Guard(GuardNode),
    GuardWithHysteresis(GuardWithHysteresisNode),
    CommittingGuard(CommittingGuardNode),
    Action(ActionNode),
    Continuous(ContinuousNode),
    StatefulContinuous(StatefulContinuousNode),
    Semaphore(SemaphoreNode),
    ScoringSelect(ScoringSelectNode),
    Repeat(RepeatNode),
    Noop,
}

impl BehaviorNode {
    pub fn tick(&mut self, situation: &mut RobotSituation) -> TickResult {
        match self {
            BehaviorNode::Select(n) => n.tick(situation),
            BehaviorNode::Sequence(n) => n.tick(situation),
            BehaviorNode::Guard(n) => n.tick(situation),
            BehaviorNode::GuardWithHysteresis(n) => n.tick(situation),
            BehaviorNode::CommittingGuard(n) => n.tick(situation),
            BehaviorNode::Action(n) => n.tick(situation),
            BehaviorNode::Continuous(n) => n.tick(situation),
            BehaviorNode::StatefulContinuous(n) => n.tick(situation),
            BehaviorNode::Semaphore(n) => n.tick(situation),
            BehaviorNode::ScoringSelect(n) => n.tick(situation),
            BehaviorNode::Repeat(n) => n.tick(situation),
            BehaviorNode::Noop => (BehaviorStatus::Success, None),
        }
    }
}

/// The default tree: do nothing, succeed.
impl Default for BehaviorNode {
    fn default() -> Self {
        BehaviorNode::Noop
    }
}

pub fn noop_node() -> BehaviorNode {
    BehaviorNode::Noop
}
