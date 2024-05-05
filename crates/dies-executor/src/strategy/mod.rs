use std::collections::HashMap;

use dies_core::{PlayerData, WorldData};

use crate::control::{PlayerControlInput, PlayerInputs};

pub trait Strategy: Send {
    fn update(&mut self, world: &WorldData) -> PlayerInputs;
}

pub trait Role: Send {
    fn update(&mut self, player_data: &PlayerData, world: &WorldData) -> PlayerControlInput;
}

pub struct AdHocStrategy {
    roles: HashMap<u32, Box<dyn Role>>,
    unassigned_roles: Vec<Box<dyn Role>>,
}

impl AdHocStrategy {
    pub fn new() -> Self {
        AdHocStrategy {
            roles: HashMap::new(),
            unassigned_roles: Vec::new(),
        }
    }

    pub fn add_role_with_id(&mut self, id: u32, role: Box<dyn Role>) {
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
            tracing::warn!("Not enough players to assign all roles");
        }

        let mut inputs = PlayerInputs::new();
        for (id, role) in self.roles.iter_mut() {
            if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
                let input = role.update(player_data, world);
                inputs.insert(*id, input);
            } else {
                tracing::error!("No detetion data for player #{id} with active role");
            }
        }
        inputs
    }
}
