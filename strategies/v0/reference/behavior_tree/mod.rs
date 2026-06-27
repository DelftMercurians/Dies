mod argument;
mod bt_core;
pub mod bt_node;
mod game_context;
pub mod role_assignment;

use std::sync::Arc;

pub(self) use argument::*;
pub(self) use bt_core::BehaviorStatus;
pub(self) use bt_node::*;

pub use argument::Argument;
pub use bt_core::*;
pub use game_context::GameContext;
pub use role_assignment::{Role, RoleAssignmentProblem, RoleAssignmentSolver, RoleBuilder};

// pub type BtCallback<TRet> = fn(&RobotSituation) -> TRet;
pub trait BtCallback<TRet>: Fn(&RobotSituation) -> TRet + Send + Sync + 'static {}
impl<TRet, F> BtCallback<TRet> for F where F: Fn(&RobotSituation) -> TRet + Send + Sync + 'static {}

pub type Strategy = fn(&mut GameContext) -> ();
