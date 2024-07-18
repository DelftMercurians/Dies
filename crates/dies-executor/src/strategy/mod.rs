mod adhoc;
pub mod attack_strat;
pub mod free_kick;
pub mod kickoff;
pub mod penalty_kick;
mod task;
pub mod test_strat;

use crate::{
    roles::{Role, RoleCtx},
    PlayerControlInput, PlayerInputs,
};
pub use adhoc::AdHocStrategy;
use dies_core::{PlayerId, WorldData};
use std::collections::HashMap;

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

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role>;
}
