mod task;
pub mod kickoff;
pub mod penalty_kick;
pub mod free_kick;

use std::collections::HashMap;

pub struct StrategyCtx<'a> {
    pub world: &'a WorldData,
}

pub trait Strategy: Send {
    /// Called when the strategy is entered (activated), before the first update.
    ///
    /// This method may be called multiple times during the lifetime of the strategy.
    fn on_enter(&mut self, _ctx: StrategyCtx) {}
    /// Update the strategy and return the inputs for all players.
    fn update(&mut self, world: &WorldData) -> PlayerInputs;

    /// Get the role type of a player, if any.
    fn get_role_type(&self, player_id: PlayerId) -> Option<RoleType>;
}

pub struct AdHocStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    unassigned_roles: Vec<Box<dyn Role>>,
    skill_map: HashMap<String, Box<dyn Skill>>,
}

impl Default for AdHocStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl AdHocStrategy {
    pub fn new() -> Self {
        AdHocStrategy {
            roles: HashMap::new(),
            unassigned_roles: Vec::new(),
            skill_map: HashMap::new(),
        }
    }

    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }

    /// Update the strategy with the current world state. Called once per frame.
    ///
    /// In this method, the strategy should assign and update roles, which will be used
    /// by the team controller to produce player inputs.
    fn update(&mut self, ctx: StrategyCtx);

    /// Get the roles assigned to players.
    fn get_roles(&mut self) -> &mut HashMap<PlayerId, Box<dyn Role>>;
}
