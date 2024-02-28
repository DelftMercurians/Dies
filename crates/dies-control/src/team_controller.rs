use std::collections::HashMap;

use dies_core::{PlayerCmd, PlayerData};
use nalgebra::Vector2;

use crate::player_controller::PlayerController;

pub struct TeamController {
    players: HashMap<u32, PlayerController>,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new() -> Self {
        Self {
            players: HashMap::new(),
        }
    }

    /// Set the target position for the player with the given ID.
    pub fn set_target_pos(&mut self, id: u32, setpoint: Vector2<f32>) {
        if let Some(player) = self.players.get_mut(&id) {
            player.set_target_pos(setpoint);
        }
    }

    /// Update the controllers with the current state of the players.
    pub fn update(&mut self, state: &Vec<PlayerData>) -> Vec<PlayerCmd> {
        state
            .iter()
            .map(|s| {
                self.players
                    .entry(s.id)
                    .or_insert_with(|| PlayerController::new(s.id))
                    .update(s)
            })
            .collect()
    }
}
