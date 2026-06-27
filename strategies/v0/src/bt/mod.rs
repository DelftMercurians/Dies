//! A self-contained behavior-tree runtime for the v0 strategy.
//!
//! This is the old in-executor `behavior_tree_api`, lifted into the strategy
//! binary and re-targeted at the IPC framework: nodes tick against a
//! [`RobotSituation`] built from the [`WorldSnapshot`](dies_strategy_protocol::WorldSnapshot)
//! and produce [`SkillCommand`](dies_strategy_protocol::SkillCommand)s (instead of
//! the executor's internal `PlayerControlInput`). The fine-grained tree
//! visualization the original engine emitted is dropped — only the control logic
//! is kept.

pub mod argument;
pub mod game_context;
pub mod nodes;
pub mod role_assignment;
pub mod runtime;
pub mod situation;

pub use argument::Argument;
pub use game_context::GameContext;
pub use nodes::*;
pub use role_assignment::{Role, RoleAssignmentProblem, RoleAssignmentSolver, RoleBuilder};
pub use runtime::BtRuntime;
pub use situation::{BehaviorStatus, BtContext, RobotSituation, ShootTarget};

/// A closure evaluated against a `RobotSituation` — the type of every tree
/// parameter, scorer, and guard condition.
pub trait BtCallback<TRet>: Fn(&RobotSituation) -> TRet + Send + Sync + 'static {}
impl<TRet, F> BtCallback<TRet> for F where F: Fn(&RobotSituation) -> TRet + Send + Sync + 'static {}

/// A strategy is a function that declares roles on the `GameContext` each frame.
pub type Strategy = fn(&mut GameContext);
