use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    fmt::format,
    time::Instant,
};

use crate::{
    roles::{RoleCtx, Skill, SkillState},
    strategy::{Strategy, StrategyCtx},
    PlayerControlInput,
};

use super::{
    force_field::compute_force,
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    rrt::find_path,
    rvo::velocity_obstacle_update,
    vo::{compute_velocity_constraints, Obstacle},
};
use dies_core::{
    debug_circle_stroke, debug_line, debug_string, debug_value, ControllerSettings, DebugColor,
    GameState, PlayerData, PlayerId, Vector2,
};
use dies_core::{PlayerCmd, WorldData};
use dodgy_2d::{Agent, AvoidanceOptions};
use serde::de;

#[derive(Default)]
struct RoleState {
    skill_map: HashMap<String, SkillState>,
}

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    strategy: Box<dyn Strategy>,
    role_states: HashMap<PlayerId, RoleState>,
    settings: ControllerSettings,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(strategy: Box<dyn Strategy>, settings: &ControllerSettings) -> Self {
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
            }
        }

        let strategy_ctx = StrategyCtx { world: &world_data };
        self.strategy.update(strategy_ctx);
        let roles = self.strategy.get_roles();
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

        let opponent_obstacles = world_data
            .opp_players
            .iter()
            .map(|p| {
                Cow::Owned::<dodgy_2d::Obstacle>(velocity_aligned_bounding_box(
                    p.position.xy(),
                    p.velocity.xy(),
                    100.0,
                    world_data.dt,
                ))
            })
            .collect::<Vec<_>>();

        // draw obstacles
        // for (obs_idx, obs) in opponent_obstacles.iter().enumerate() {
        //     match obs.as_ref() {
        //         dodgy_2d::Obstacle::Closed { vertices } => {
        //             for i in 0..vertices.len() {
        //                 let j = (i + 1) % vertices.len();
        //                 let a = Vector2::new(vertices[i].x as f64, vertices[i].y as f64);
        //                 let b = Vector2::new(vertices[j].x as f64, vertices[j].y as f64);
        //                 debug_line(format!("obstacle-{}-{}", obs_idx, i), a, b, DebugColor::Red);
        //             }
        //         }
        //         dodgy_2d::Obstacle::Open { vertices } => {
        //             for i in 0..vertices.len() {
        //                 if i == vertices.len() - 1 {
        //                     break;
        //                 }
        //                 let j = (i + 1) % vertices.len();
        //                 let a = Vector2::new(vertices[i].x as f64, vertices[i].y as f64);
        //                 let b = Vector2::new(vertices[j].x as f64, vertices[j].y as f64);
        //                 debug_line(format!("obstacle-{}-{}", obs_idx, i), a, b, DebugColor::Red);
        //             }
        //         }
        //     }
        // }

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

                // if id.as_u32() == 0 {
                //     let obstacles = vec![
                //         // Obstacle::Circular {
                //         //     center: Vector2::new(0.0, 0.0),
                //         //     velocity: Vector2::new(0.0, 0.0),
                //         //     radius: 100.0,
                //         // },
                //         Obstacle::Rectangular {
                //             center: Vector2::new(1000.0, 1000.0),
                //             width: 200.0,
                //             height: 200.0,
                //         },
                //     ];
                //     // draw obstacles
                //     obstacles
                //         .iter()
                //         .enumerate()
                //         .for_each(|(idx, obs)| match obs {
                //             Obstacle::Circular { center, radius, .. } => {
                //                 debug_circle_stroke(
                //                     format!("obstacle-{}", idx),
                //                     *center,
                //                     *radius,
                //                     DebugColor::Red,
                //                 );
                //             }
                //             Obstacle::Rectangular {
                //                 center,
                //                 width,
                //                 height,
                //             } => {
                //                 let half_width = *width / 2.0;
                //                 let half_height = *height / 2.0;
                //                 let vertices = vec![
                //                     Vector2::new(center.x - half_width, center.y - half_height),
                //                     Vector2::new(center.x + half_width, center.y - half_height),
                //                     Vector2::new(center.x + half_width, center.y + half_height),
                //                     Vector2::new(center.x - half_width, center.y + half_height),
                //                 ];
                //                 for i in 0..vertices.len() {
                //                     let j = (i + 1) % vertices.len();
                //                     debug_line(
                //                         format!("obstacle-{}-{}", idx, i),
                //                         vertices[i],
                //                         vertices[j],
                //                         DebugColor::Red,
                //                     );
                //                 }
                //             }
                //         });
                //     compute_velocity_constraints(
                //         player_data,
                //         &world_data.own_players,
                //         &obstacles,
                //         0.5,
                //     )
                //     .iter()
                //     .enumerate()
                //     .for_each(|(idx, constraint)| {
                //         let center = player_data.position + constraint.apex;
                //         let left = center + constraint.left * 1000.0;
                //         let right = center + constraint.right * 1000.0;
                //         debug_line(
                //             format!("constraint-{}-left", idx),
                //             center,
                //             left,
                //             DebugColor::Purple,
                //         );
                //         debug_line(
                //             format!("constraint-{}-right", idx),
                //             center,
                //             right,
                //             DebugColor::Purple,
                //         );
                //     });
                // }

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                if !is_manual {
                    let start = Instant::now();
                    let vel = velocity_obstacle_update(
                        player_data,
                        &controller.target_velocity(),
                        all_players.as_slice(),
                        &vec![],
                        super::rvo::VelocityObstacleType::VO,
                    );
                    controller.update_target_velocity_with_avoidance(vel);
                    let end = Instant::now();
                    debug_value(
                        format!("rvo-time-{}", id),
                        end.duration_since(start).as_millis() as f64,
                    );
                    // let position = player_data.position;
                    // let agent = agent_from_player(player_data);
                    // let obstacle_margin = 50.0;
                    // let time_horizon = world_data.dt * 10.0;
                    // let obstacle_time_horizon = world_data.dt * 4.0;
                    // let query_dist = self.settings.max_velocity * time_horizon + 200.0;
                    // let neighbors = world_data
                    //     .own_players
                    //     .iter()
                    //     .filter_map(|p| {
                    //         if p.id != id && (p.position - position).norm() < query_dist {
                    //             Some(agent_from_player(p))
                    //         } else {
                    //             None
                    //         }
                    //     })
                    //     .map(Cow::Owned)
                    //     .collect::<Vec<_>>();
                    // let target_vel = controller.target_velocity();
                    // let target_vel = dodgy_2d::Vec2 {
                    //     x: target_vel.x as f32,
                    //     y: target_vel.y as f32,
                    // };
                    // let avoidance_velocity = agent.compute_avoiding_velocity(
                    //     // Neighbors - other players
                    //     neighbors.as_slice(),
                    //     // Obstacles
                    //     opponent_obstacles.as_slice(),
                    //     target_vel,
                    //     self.settings.max_velocity as f32,
                    //     world_data.dt as f32,
                    //     &AvoidanceOptions {
                    //         obstacle_margin,
                    //         time_horizon: time_horizon as f32,
                    //         obstacle_time_horizon: obstacle_time_horizon as f32,
                    //     },
                    // );
                    // let avoidance_velocity =
                    //     Vector2::new(avoidance_velocity.x as f64, avoidance_velocity.y as f64);
                    // controller.update_target_velocity_with_avoidance(avoidance_velocity);
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

fn agent_from_player(player: &PlayerData) -> dodgy_2d::Agent {
    dodgy_2d::Agent {
        position: dodgy_2d::Vec2::new(player.position.x as f32, player.position.y as f32),
        velocity: dodgy_2d::Vec2::new(player.velocity.x as f32, player.velocity.y as f32),
        radius: 100.0,
        avoidance_responsibility: player.velocity.norm() as f32,
    }
}

fn velocity_aligned_bounding_box(
    position: Vector2,
    velocity: Vector2,
    radius: f64,
    dt: f64,
) -> dodgy_2d::Obstacle {
    let position = dodgy_2d::Vec2::new(position.x as f32, position.y as f32);
    if velocity.magnitude() == 0.0 {
        let radius = radius as f32;
        return dodgy_2d::Obstacle::Closed {
            vertices: vec![
                position + dodgy_2d::Vec2::new(radius, radius),
                position + dodgy_2d::Vec2::new(-radius, radius),
                position + dodgy_2d::Vec2::new(-radius, -radius),
                position + dodgy_2d::Vec2::new(radius, -radius),
            ],
        };
    }

    let direction = velocity.normalize();
    let perpendicular = Vector2::new(-direction.y, direction.x);
    let half_width = radius * perpendicular;
    let half_length = 0.5 * velocity * dt + radius * direction;

    let half_width = dodgy_2d::Vec2::new(half_width.x as f32, half_width.y as f32);
    let half_length = dodgy_2d::Vec2::new(half_length.x as f32, half_length.y as f32);

    dodgy_2d::Obstacle::Closed {
        vertices: vec![
            position + half_width + half_length,
            position - half_width + half_length,
            position - half_width - half_length,
            position + half_width - half_length,
        ],
    }
}
