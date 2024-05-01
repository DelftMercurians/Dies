use std::collections::{HashMap, HashSet};

use crate::player_controller::{self, KickerControlInput, PlayerControlInput, PlayerController};
use dies_core::GameState;
use dies_core::{PlayerCmd, WorldData};
use nalgebra::Vector2;
use std::cell::RefCell;
use std::rc::Rc;

/// Everyone stops, notice that this only interrupts the players, so if the game
/// recovers the players will head to their original goal.
pub struct HaltController;

impl HaltController {
    pub fn new() -> Self {
        Self
    }

    pub fn update(&mut self, world: &WorldData) -> Vec<PlayerControlInput> {
        world
            .own_players
            .iter()
            .map(|p| PlayerControlInput::new(p.id))
            .collect()
    }
}

pub struct TeamController {
    player_controllers: HashMap<u32, PlayerController>,
    halt_controller: HaltController,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new() -> Self {
        let players = HashMap::new();
        Self {
            halt_controller: HaltController::new(),
            player_controllers: players,
        }
    }

    /// Update the controllers with the current state of the players.
    pub fn update(&mut self, world_data: WorldData) {
        // Ensure there is a player controller for every ID
        let detected_ids: HashSet<_> = world_data.own_players.iter().map(|p| p.id).collect();
        for id in detected_ids.iter() {
            if !self.player_controllers.contains_key(id) {
                self.player_controllers
                    .insert(*id, PlayerController::new(*id));
            }
        }

        let mut inputs = match world_data.current_game_state {
            GameState::Halt | GameState::Timeout => self.halt_controller.update(&world_data),
            _ => Vec::new(),
        };

        // If in a stop state, override the inputs
        if world_data.current_game_state == GameState::Stop {
            inputs = stop_override(world_data.clone(), inputs);
        }

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let input = inputs
                    .iter()
                    .find(|i| i.id == controller.id())
                    .cloned()
                    .unwrap_or_else(|| PlayerControlInput::new(controller.id()));

                controller.update(player_data, input);
            } else {
                controller.increment_frames_missings();
            }
        }
    }

    /// Get the currently active commands for the players.
    pub fn commands(&self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values()
            .map(|c| c.command())
            .collect()
    }
}

/// Override the inputs to comply with the stop state.
fn stop_override(
    world_data: WorldData,
    inputs: Vec<PlayerControlInput>,
) -> Vec<PlayerControlInput> {
    let ball_pos = world_data.ball.as_ref().map(|b| b.position.xy());
    let ball_vel = world_data.ball.as_ref().map(|b| b.velocity.xy());
    inputs
        .iter()
        .map(|input| {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == input.id)
                .expect("Player not found in world data");

            let mut new_input = input.clone();

            // Cap speed at 1.5m/s
            new_input.velocity = input.velocity.cap_magnitude(1.5);

            // If the player is less than 500mm from the ball, set the goal to the point 500mm away
            // from the ball, in the opposite direction of the ball's speed.
            if let (Some(ball_pos), Some(ball_vel)) = (ball_pos, ball_vel) {
                let dist = (player_data.position - ball_pos).norm();
                if dist < 500.0 {
                    let goal = ball_pos - ball_vel.normalize() * 500.0;
                    new_input.position = Some(goal);
                }
            }

            // Stop dribbler
            new_input.dribbling_speed = 0.0;

            // Disable kick
            new_input.kicker = KickerControlInput::Disarm;

            new_input
        })
        .collect()
}
