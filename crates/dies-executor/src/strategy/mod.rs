mod task;
mod adhoc;
pub mod kickoff;
pub mod penalty_kick;
pub mod free_kick;

use std::collections::HashMap;
pub use adhoc::AdHocStrategy;
use crate::roles::Role;
use dies_core::{PlayerId, WorldData};

pub struct StrategyCtx<'a> {
    pub world: &'a WorldData,
}

pub trait Strategy: Send {
    /// Called when the strategy is entered (activated), before the first update.
    ///
    /// This method may be called multiple times during the lifetime of the strategy.
    fn on_enter(&mut self, _ctx: StrategyCtx) {}

    /// Update the strategy with the current world state. Called once per frame.
    ///
    /// In this method, the strategy should assign and update roles, which will be used
    /// by the team controller to produce player inputs.
    fn update(&mut self, ctx: StrategyCtx);

    /// Get the roles assigned to players.
    fn get_roles(&mut self) -> &mut HashMap<PlayerId, Box<dyn Role>>;
}
