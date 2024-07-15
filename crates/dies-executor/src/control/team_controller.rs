use crate::{strategy::Strategy, PlayerControlInput};
use std::collections::{HashMap, HashSet};

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    rvo::velocity_obstacle_update,
};
use crate::{
    roles::{RoleCtx, SkillState},
    strategy::StrategyCtx,
};
use dies_core::{ControllerSettings, GameState, PlayerId};
use dies_core::{PlayerCmd, WorldData};

#[derive(Default)]
struct RoleState {
    skill_map: HashMap<String, SkillState>,
}

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    strategy: HashMap<GameState, Box<dyn Strategy>>,
    role_states: HashMap<PlayerId, RoleState>,
    settings: ControllerSettings,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(
        strategy: HashMap<GameState, Box<dyn Strategy>>,
        settings: &ControllerSettings,
    ) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            strategy,
            role_states: HashMap::new(),
            settings: settings.clone(),
        };
        team.update_controller_settings(settings);
        team
    }

    pub fn update_controller_settings(&mut self, settings: &ControllerSettings) {
        for controller in self.player_controllers.values_mut() {
            controller.update_settings(settings);
        }
        self.settings = settings.clone();
    }

    /// Update the controllers with the current state of the players.
    pub fn update(
        &mut self,
        world_data: WorldData,
        manual_override: HashMap<PlayerId, PlayerControlInput>,
    ) {
        // Ensure there is a player controller for every ID
        let detected_ids: HashSet<_> = world_data.own_players.iter().map(|p| p.id).collect();
        for id in detected_ids.iter() {
            if !self.player_controllers.contains_key(id) {
                self.player_controllers
                    .insert(*id, PlayerController::new(*id, &self.settings));
                if self.player_controllers.len() == 1 {
                    self.player_controllers
                        .get_mut(id)
                        .unwrap()
                        .set_gate_keeper();
                }
            }
        }
        let state = world_data.current_game_state.game_state;

        let strategy = if let Some(strategy) = self.strategy.get_mut(&state) {
            strategy
        } else {
            log::warn!("No strategy found for game state {:?}", state);
            return;
        };

        let strategy_ctx = StrategyCtx { world: &world_data };
        strategy.update(strategy_ctx);
        let roles = strategy.get_roles();
        let mut inputs = roles
            .iter_mut()
            .fold(PlayerInputs::new(), |mut inputs, (id, role)| {
                let player_data = world_data
                    .own_players
                    .iter()
                    .find(|p| p.id == *id)
                    .expect("Player not found in world data");

                let role_state = self.role_states.entry(*id).or_default();
                let role_ctx = RoleCtx::new(player_data, &world_data, &mut role_state.skill_map);
                let new_input = role.update(role_ctx);
                inputs.insert(*id, new_input);
                inputs
            });

        // If in a stop state, override the inputs
        if world_data.current_game_state.game_state == GameState::Stop {
            inputs = stop_override(&world_data, inputs);
        }

        let all_players = world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
            .collect::<Vec<_>>();

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let id = controller.id();
                let default_input = inputs.player(id);
                let input = manual_override.get(&id).unwrap_or(&default_input);
                controller.update(player_data, &world_data, input, world_data.dt);

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                if !is_manual {
                    let vel = velocity_obstacle_update(
                        player_data,
                        &controller.target_velocity(),
                        all_players.as_slice(),
                        &[],
                        &world_data.player_model,
                        super::rvo::VelocityObstacleType::VO,
                    );
                    controller.update_target_velocity_with_avoidance(vel);
                }
            } else {
                controller.increment_frames_misses();
            }
        }
    }

    /// Get the currently active commands for the players.
    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values_mut()
            .map(|c| c.command())
            .collect()
    }
}

/// Override the inputs to comply with the stop state.
fn stop_override(world_data: &WorldData, inputs: PlayerInputs) -> PlayerInputs {
    let ball_pos = world_data.ball.as_ref().map(|b| b.position.xy());
    let ball_vel = world_data.ball.as_ref().map(|b| b.velocity.xy());
    inputs
        .iter()
        .map(|(id, input)| {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == *id)
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

            (*id, new_input)
        })
        .collect()
}
