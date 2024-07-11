use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
};

use log::log;
use crate::{strategy::Strategy, PlayerControlInput};
use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
};
use dies_core::{ControllerSettings, GameState, PlayerId, Vector2};
use dies_core::{PlayerCmd, WorldData};
use crate::strategy::AdHocStrategy;
use crate::strategy::kickoff::KickoffStrategy;
use dodgy_2d::{Agent, AvoidanceOptions};

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    strategy: HashMap<GameState, Box<dyn Strategy>>,
    settings: ControllerSettings,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(strategy: HashMap<GameState, Box<dyn Strategy>>, settings: &ControllerSettings) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            strategy,
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
                    self.player_controllers.get_mut(id).unwrap().set_gate_keeper();
                }
            }
        }
        let state = world_data.current_game_state.game_state;

        let mut inputs = if let Some(strategy) = self.strategy.get_mut(&state) {
            strategy
        } else {
            log::warn!("No strategy found for game state {:?}", state);
            return;
        }.update(&world_data);

        // If in a stop state, override the inputs
        if world_data.current_game_state.game_state == GameState::Stop {
            inputs = stop_override(world_data.clone(), inputs);
        }

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let default_input = inputs.player(controller.id());
                let input = manual_override
                    .get(&controller.id())
                    .unwrap_or(&default_input);
                controller.update(player_data, input, world_data.dt);
            } else {
                controller.increment_frames_misses();
            }
        }

        // Compute cooperative avoidance velocities
        let own_agents = world_data
            .own_players
            .iter()
            .map(|p| {
                (
                    p.id,
                    Cow::Owned::<Agent>(Agent {
                        position: dodgy_2d::Vec2 {
                            x: p.position.x as f32,
                            y: p.position.y as f32,
                        },
                        velocity: dodgy_2d::Vec2 {
                            x: p.velocity.x as f32,
                            y: p.velocity.y as f32,
                        },
                        // Not sure why, but a higher radius seems to work better
                        radius: 200.0,
                        avoidance_responsibility: if self
                            .player_controllers
                            .get(&p.id)
                            .unwrap()
                            .target_velocity()
                            .norm()
                            > 0.0
                        {
                            1.0
                        } else {
                            0.0
                        },
                    }),
                )
            })
            .collect::<Vec<_>>();
        let opp_agents = world_data
            .opp_players
            .iter()
            .map(|p| {
                Cow::Owned::<Agent>(Agent {
                    position: dodgy_2d::Vec2 {
                        x: p.position.x as f32,
                        y: p.position.y as f32,
                    },
                    velocity: dodgy_2d::Vec2 {
                        x: p.velocity.x as f32,
                        y: p.velocity.y as f32,
                    },
                    // Not sure why, but a higher radius seems to work better
                    radius: 200.0,
                    avoidance_responsibility: 0.,
                })
            })
            .collect::<Vec<_>>();

        for (id, agent) in own_agents.iter() {
            let obstacles = world_data
                .get_obstacles_for_player(self.strategy.get_role_type(*id).unwrap_or_default());

            let obstacles = obstacles
                .iter()
                .map(|o| Cow::Borrowed(o))
                .collect::<Vec<_>>();

            let neighbors = own_agents
                .iter()
                .filter(|(other_id, _)| *other_id != *id)
                .map(|(_, a)| a)
                .chain(opp_agents.iter())
                .cloned()
                .collect::<Vec<_>>();

            let player_controller = self
                .player_controllers
                .get_mut(id)
                .expect("Player controller not found");
            let target_vel = player_controller.target_velocity();
            let target_vel = dodgy_2d::Vec2 {
                x: target_vel.x as f32,
                y: target_vel.y as f32,
            };

            if agent.avoidance_responsibility == 0.0 {
                continue;
            }

            let avoidance_velocity = agent.compute_avoiding_velocity(
                // Neighbors - other players
                neighbors.as_slice(),
                // Obstacles
                &obstacles,
                target_vel,
                self.settings.max_velocity as f32,
                world_data.dt as f32,
                &AvoidanceOptions {
                    obstacle_margin: 100.0,
                    time_horizon: 3.0,
                    obstacle_time_horizon: 3.0,
                },
            );

            let avoidance_velocity =
                Vector2::new(avoidance_velocity.x as f64, avoidance_velocity.y as f64);
            player_controller.update_target_velocity_with_avoidance(avoidance_velocity);
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
fn stop_override(world_data: WorldData, inputs: PlayerInputs) -> PlayerInputs {
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
