mod Task;
pub mod kickoff;
mod penalty_kick;

use std::collections::HashMap;

use dies_core::{PlayerId, WorldData};

use crate::{
    control::PlayerInputs,
    roles::{Role, RoleCtx, Skill},
};

pub trait Strategy: Send {
    fn update(&mut self, world: &WorldData) -> PlayerInputs;
}

pub struct AdHocStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    unassigned_roles: Vec<Box<dyn Role>>,
    skill_map: HashMap<String, Box<dyn Skill>>,
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

    /// Add a role to the list of unassigned roles. On the next update, the role will be
    /// to the first available player.
    pub fn add_role(&mut self, role: Box<dyn Role>) {
        self.unassigned_roles.push(role);
    }
}

impl Strategy for AdHocStrategy {

    fn update(&mut self, world: &WorldData) -> PlayerInputs {
        // Assign roles to players
        for player_data in world.own_players.iter() {
            if !self.roles.contains_key(&player_data.id) {
                if let Some(role) = self.unassigned_roles.pop() {
                    self.roles.insert(player_data.id, role);
                }
            }
        }
        if self.unassigned_roles.len() > 0 {
            log::warn!("Not enough players to assign all roles");
        }


        let mut inputs = PlayerInputs::new();
        for (id, role) in self.roles.iter_mut() {
            if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
                let ctx = RoleCtx::new(player_data, world, &mut self.skill_map);
                let input = role.update(ctx);
                inputs.insert(*id, input);
            } else {
                log::error!("No detetion data for player #{id} with active role");
            }
        }
        inputs
    }
}
