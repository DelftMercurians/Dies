mod avoidance;
mod joint_skill_executor;
mod pass_coordinator;
mod passing;
mod path_follower;
mod player_controller;
mod player_input;
pub mod skill_executor;
mod team_context;
mod team_controller;
mod yaw_control;

pub use team_context::*;

// `ObstacleSet` is already part of `control`'s surface via the public
// `SkillContext::obstacles` field; name it so out-of-`control` skill tests can
// build a `SkillContext`.
pub use avoidance::ObstacleSet;
// Shared world/player/context fixtures for skill unit tests across the crate.
#[cfg(test)]
pub(crate) use pass_coordinator::test_support;

pub use joint_skill_executor::JointSkillExecutor;
pub use pass_coordinator::{PassContext, PassCoordinator, PassPhase, PassTickOutput};
pub use passing::*;
pub use player_input::*;
pub use skill_executor::{
    ExecutableSkill, SkillContext, SkillExecutor, SkillProgress, SkillResult,
};
pub use team_controller::{StrategyInput, TeamController};
