use std::collections::HashMap;

use dies_core::{PlayerId, WorldData};

use crate::roles::Role;

pub struct StrategyCtx<'a> {
    pub world: &'a WorldData,
}

pub trait Strategy: Send {
    /// Update the strategy with the current world state.
    fn update(&mut self, ctx: StrategyCtx);

    /// Get the roles assigned to players.
    fn get_roles(&mut self) -> &mut HashMap<PlayerId, Box<dyn Role>>;
}

pub struct AdHocStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    unassigned_roles: Vec<Box<dyn Role>>,
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
        }
    }

    /// Add a role to a specific player.
    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }

    /// Add a role to the list of unassigned roles. On the next update, the role will be
    /// to the first available player.
    pub fn add_role(&mut self, role: Box<dyn Role>) {
        self.unassigned_roles.push(role);
    }
}

impl Strategy for AdHocStrategy {
    fn update(&mut self, ctx: StrategyCtx) {
        // Assign roles to players
        for player_data in ctx.world.own_players.iter() {
            if let std::collections::hash_map::Entry::Vacant(e) = self.roles.entry(player_data.id) {
                if let Some(role) = self.unassigned_roles.pop() {
                    e.insert(role);
                }
            }
        }
        if !self.unassigned_roles.is_empty() {
            log::warn!("Not enough players to assign all roles");
        }
    }

    fn get_roles(&mut self) -> &mut HashMap<PlayerId, Box<dyn Role>> {
        &mut self.roles
    }
}
