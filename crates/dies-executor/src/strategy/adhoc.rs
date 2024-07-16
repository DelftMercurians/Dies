use super::Strategy;
use super::StrategyCtx;
use crate::roles::Role;
use dies_core::PlayerId;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// A strategy that assigns roles to players based on a predefined list. Useful for
/// testing.
///
/// This strategy will reassign roles every time it is entered.
///
/// # Example
///
/// ```ignore
/// let mut strategy = AdHocStrategy::new();
///
/// // This role will be assigned to player 0 if there is a player with that ID,
/// // otherwise nothing will happen.
/// strategy.add_role_with_id(0, Box::new(Attacker::new()));
///
/// // This role will be assigned to the first available player.
/// strategy.add_role(Box::new(Defender::new()));
/// ```
pub struct AdHocStrategy {
    pub(crate) roles: HashMap<PlayerId, Box<dyn Role>>,
    pub(crate) unassigned_roles: Vec<Box<dyn Role>>,
    fixed_roles: HashMap<PlayerId, Box<dyn Role>>,
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
            fixed_roles: HashMap::new(),
        }
    }

    /// Add a role to a specific player. If the player does not exist, the role will
    /// never be updated so effectively it will not be assigned.
    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.fixed_roles.insert(id, role);
    }

    /// Add a role to the list of unassigned roles. On the next update, the role will be
    /// to the first available player.
    pub fn add_role(&mut self, role: Box<dyn Role>) {
        self.unassigned_roles.push(role);
    }
}

impl Strategy for AdHocStrategy {
    fn on_enter(&mut self, _ctx: StrategyCtx) {
        // Clear roles
        self.roles.clear();
    }

    fn update(&mut self, ctx: StrategyCtx) {
        // Assign roles to players
        for player_data in ctx.world.own_players.iter() {
            if let Some(role) = self.fixed_roles.remove(&player_data.id) {
                self.roles.insert(player_data.id, role);
            } else if let Entry::Vacant(e) = self.roles.entry(player_data.id) {
                if let Some(role) = self.unassigned_roles.pop() {
                    e.insert(role);
                }
            }
        }
        if !self.unassigned_roles.is_empty() {
            log::warn!("Not enough players to assign all roles");
        }
    }
    
    fn update_role(&mut self, player_id: PlayerId, ctx: crate::roles::RoleCtx) -> Option<crate::PlayerControlInput> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.update(ctx))
        } else {
            None
        }
    }
}
