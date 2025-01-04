use std::collections::{HashMap, HashSet};

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
};
use crate::{skills::SkillType, team_frame::TeamFrame, Obstacle, PlayerControlInput};
use dies_core::{Angle, PlayerId, RobotCmd, Vector2, WorldFrame};
use dies_protos::ssl_gc_state::GameState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TeamCommand {
    Move {
        id: PlayerId,
        target: Vector2,
        yaw: Option<Angle>,
    },
    ApproachBall {
        id: PlayerId,
    },
    Kick {
        id: PlayerId,
    },
}

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(settings: &ExecutorSettings) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
        };
        team.update_controller_settings(settings);
        team
    }

    pub fn set_opp_goal_sign(&mut self, opp_goal_sign: f64) {
        for controller in self.player_controllers.values_mut() {
            controller.set_opp_goal_sign(opp_goal_sign);
        }
    }

    pub fn update_controller_settings(&mut self, settings: &ExecutorSettings) {
        for controller in self.player_controllers.values_mut() {
            controller.update_settings(&settings.controller_settings);
        }
        self.settings = settings.clone();
    }

    pub fn handle_team_command(&mut self, cmd: TeamCommand) {
        match cmd {
            TeamCommand::Move { id, target, yaw } => {
                if let Some(controller) = self.player_controllers.get_mut(&id) {
                    controller.activate_skill(SkillType::GoToPosition {
                        target_pos: target,
                        target_heading: yaw,
                        with_ball: false,
                        avoid_ball: false,
                    });
                }
            }
            TeamCommand::ApproachBall { id } => {
                if let Some(controller) = self.player_controllers.get_mut(&id) {
                    controller.activate_skill(SkillType::ApproachBall);
                }
            }
            TeamCommand::Kick { id } => {
                if let Some(controller) = self.player_controllers.get_mut(&id) {
                    controller.activate_skill(SkillType::Kick);
                }
            }
        }
    }

    /// Update the controllers with the current state of the players.
    pub fn update(
        &mut self,
        team_frame: TeamFrame,
        manual_override: HashMap<PlayerId, PlayerControlInput>,
    ) {
        // Ensure there is a player controller for every ID
        let detected_ids: HashSet<_> = team_frame.own_players.iter().map(|p| p.id).collect();
        for id in detected_ids.iter() {
            if !self.player_controllers.contains_key(id) {
                self.player_controllers
                    .insert(*id, PlayerController::new(*id, &self.settings));
            }
        }

        let all_players = team_frame
            .own_players
            .iter()
            .chain(team_frame.opp_players.iter())
            .collect::<Vec<_>>();

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = team_frame
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let id = controller.id();

                let role_type = Default::default(); //role_types.get(&id).cloned().unwrap_or_default();
                let obsacles = team_frame.get_obstacles_for_player(role_type);

                controller.update(
                    player_data,
                    &team_frame,
                    team_frame.dt,
                    manual_override.get(&id),
                    obsacles,
                    &all_players,
                );
            } else {
                controller.increment_frames_missed();
            }
        }
    }

    /// Get the currently active commands for the players.
    pub fn commands(&mut self) -> Vec<(PlayerId, RobotCmd)> {
        self.player_controllers
            .iter()
            .map(|(id, controller)| (id, controller.command()))
            .collect()
    }
}

/// Override the inputs to comply with the stop state.
fn comply(team_frame: &TeamFrame, inputs: PlayerInputs) -> PlayerInputs {
    if let (Some(ball), Some(field)) = (team_frame.ball.as_ref(), team_frame.field_geom.as_ref()) {
        let game_state = team_frame.game_state.game_state;
        let ball_pos = ball.position.xy();

        inputs
            .iter()
            .map(|(id, input)| {
                let player_data = team_frame
                    .own_players
                    .iter()
                    .find(|p| p.id == *id)
                    .expect("Player not found in world data");

                let mut new_input = input.clone();

                if game_state == GameState::Stop
                    || (game_state == GameState::FreeKick
                        && input.role_type != RoleType::FreeKicker)
                {
                    new_input.avoid_robots = false;
                    new_input.avoid_ball = false;
                    if game_state == GameState::Stop {
                        new_input.with_speed_limit(1300.0);
                        new_input.dribbling_speed = 0.0;
                        new_input.kicker = KickerControlInput::Disarm;
                    }

                    let min_distance = 800.0;
                    let max_radius = 4000;
                    let target = nearest_safe_pos(
                        crate::Avoid::Circle { center: ball_pos },
                        min_distance,
                        player_data.position,
                        input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    new_input.with_position(target);
                }

                if let GameState::BallPlacement(pos) = game_state {
                    let line_start = ball_pos;
                    let line_end = pos;
                    dies_core::debug_line(
                        "ball_placement",
                        line_start,
                        line_end,
                        dies_core::DebugColor::Orange,
                    );
                    dies_core::debug_cross(
                        "ball_placement_target",
                        pos,
                        dies_core::DebugColor::Orange,
                    );

                    let min_distance = 800.0;
                    let max_radius = 4000;
                    let target = ::nearest_safe_pos(
                        crate::Avoid::Line {
                            start: line_start,
                            end: line_end,
                        },
                        min_distance,
                        player_data.position,
                        input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    new_input.with_position(target);
                } else {
                    dies_core::debug_remove("ball_placement");
                    dies_core::debug_remove("ball_placement_target");
                }

                (*id, new_input)
            })
            .collect()
    } else {
        inputs
    }
}

fn get_obstacles_for_player(world: &WorldFrame, role: RoleType) -> Vec<Obstacle> {
    if let Some(field_geom) = world.field_geom {
        let field_boundary = {
            let hl = field_geom.field_length / 2.0;
            let hw = field_geom.field_width / 2.0;
            Obstacle::Rectangle {
                min: Vector2::new(
                    -hl - field_geom.boundary_width,
                    -hw - field_geom.boundary_width,
                ),
                max: Vector2::new(
                    hl + field_geom.boundary_width,
                    hw + field_geom.boundary_width,
                ),
            }
        };
        let mut obstacles = vec![field_boundary];

        // Add own defence area for non-keeper robots
        if role != RoleType::Goalkeeper {
            let lower = Vector2::new(-10_000.0, -field_geom.penalty_area_width / 2.0);
            let upper = Vector2::new(
                -field_geom.field_length / 2.0 + field_geom.penalty_area_depth + 50.0,
                field_geom.penalty_area_width / 2.0,
            );

            let defence_area = Obstacle::Rectangle {
                min: lower,
                max: upper,
            };
            obstacles.push(defence_area);
        }

        // Add opponent defence area for all robots
        let lower = Vector2::new(
            field_geom.field_length / 2.0 - field_geom.penalty_area_depth - 50.0,
            -field_geom.penalty_area_width / 2.0,
        );
        let upper = Vector2::new(10_0000.0, field_geom.penalty_area_width / 2.0);
        let defence_area = Obstacle::Rectangle {
            min: lower,
            max: upper,
        };
        obstacles.push(defence_area);

        match self.game_state.game_state {
            GameState::Stop => {
                // Add obstacle to prevent getting close to the ball
                if let Some(ball) = &self.ball {
                    obstacles.push(Obstacle::Circle {
                        center: ball.position.xy(),
                        radius: STOP_BALL_AVOIDANCE_RADIUS,
                    });
                }
            }
            GameState::Kickoff | GameState::PrepareKickoff => match role {
                RoleType::KickoffKicker => {}
                _ => {
                    // Add center circle for non kicker robots
                    obstacles.push(Obstacle::Circle {
                        center: Vector2::zeros(),
                        radius: field_geom.center_circle_radius,
                    });
                }
            },
            GameState::BallPlacement(_) => {}
            GameState::PreparePenalty => {}
            GameState::FreeKick => {}
            GameState::Penalty => {}
            GameState::PenaltyRun => {}
            GameState::Run | GameState::Halt | GameState::Timeout | GameState::Unknown => {
                // Nothing to do
            }
        };

        obstacles
    } else {
        vec![]
    }
}
