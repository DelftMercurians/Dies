use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use dies_core::{
    ControllerMode, ExecutorSettings, GameState, Obstacle, PlayerCmd, PlayerCmdUntransformer,
    PlayerId, RoleType, SideAssignment, TeamColor, TeamData, Vector2,
};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use std::sync::Arc;

use super::{
    ilqr::IlqrController,
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    skill_executor::{SkillContext, SkillExecutor},
    team_context::TeamContext,
};
use crate::PlayerControlInput;

/// Input for the team controller from the strategy host.
#[derive(Debug, Clone, Default)]
pub struct StrategyInput {
    /// Skill commands per player (None means continue previous skill).
    pub skill_commands: HashMap<PlayerId, Option<SkillCommand>>,
    /// Role names per player for UI display.
    pub player_roles: HashMap<PlayerId, String>,
}

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,

    // Skill execution for strategy-controlled path
    skill_executor: SkillExecutor,
    /// Current strategy input (skill commands and roles).
    strategy_input: StrategyInput,

    ilqr_controller: IlqrController,

    removing_players: HashSet<PlayerId>,

    warmup_timer: Option<Instant>,
    warmup_done: bool,

    /// When false, game-state-driven compliance mutations are skipped entirely
    /// — used in test mode so scenarios control safety themselves.
    pub(crate) comply_enabled: bool,

    // per-robot flags
    avoid_goal_area_flags: HashMap<PlayerId, bool>,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(settings: &ExecutorSettings, _team_color: TeamColor) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
            skill_executor: SkillExecutor::new(),
            strategy_input: StrategyInput::default(),
            ilqr_controller: IlqrController::load_or_insert("x"),
            removing_players: HashSet::new(),
            avoid_goal_area_flags: HashMap::new(),
            warmup_timer: None,
            warmup_done: false,
            comply_enabled: true,
        };
        team.update_controller_settings(settings);
        team
    }

    pub fn update_controller_settings(&mut self, settings: &ExecutorSettings) {
        for controller in self.player_controllers.values_mut() {
            controller.update_settings(&settings.controller_settings);
        }
        self.settings = settings.clone();
    }

    /// Set strategy input (skill commands and roles from strategy host).
    pub fn set_strategy_input(&mut self, input: StrategyInput) {
        self.strategy_input = input;
    }

    /// Get current skill statuses for all players.
    pub fn get_skill_statuses(&self) -> HashMap<PlayerId, SkillStatus> {
        self.skill_executor.get_all_statuses()
    }

    pub fn update(
        &mut self,
        team_color: TeamColor,
        side_assignment: SideAssignment,
        team_data: TeamData,
        manual_override: HashMap<PlayerId, PlayerControlInput>,
    ) {
        let world_data = Arc::new(team_data);
        let detected_ids: HashSet<_> = world_data.own_players.iter().map(|p| p.id).collect();
        for id in detected_ids.iter() {
            if !self.player_controllers.contains_key(id) {
                self.player_controllers
                    .insert(*id, PlayerController::new(*id, &self.settings));
                // Initialize avoid_goal_area flag to true for new players
                self.avoid_goal_area_flags.insert(*id, true);
            }
        }
        let team_context = TeamContext::new(team_color, side_assignment);

        // Handle warmup period
        if let Some(warmup_timer) = self.warmup_timer {
            if warmup_timer.elapsed() > Duration::from_secs(1) {
                self.warmup_done = true;
                self.warmup_timer = None;
            }
        }
        if !self.warmup_done || (world_data.field_geom.is_none()) {
            if self.warmup_timer.is_none() {
                self.warmup_timer = Some(Instant::now());
            }

            // Update player controllers with default input during warmup
            for controller in self.player_controllers.values_mut() {
                let player_data = world_data
                    .own_players
                    .iter()
                    .find(|p| p.id == controller.id());
                if let Some(player_data) = player_data {
                    let player_context = team_context.player_context(controller.id());
                    let default_input = PlayerControlInput::default();
                    let input_to_use = manual_override
                        .get(&controller.id())
                        .unwrap_or(&default_input);
                    controller.update(
                        player_data,
                        &world_data,
                        input_to_use,
                        world_data.dt,
                        false,
                        Vec::new(),
                        &[],
                        &player_context,
                        true,
                        0.0,
                        true,
                    );
                }
            }

            return;
        }

        // Get active robots
        let active_robots: Vec<PlayerId> = world_data
            .own_players
            .iter()
            .map(|p| p.id)
            .filter(|id| !self.removing_players.contains(id))
            .collect();

        // Get player inputs from strategy path
        let (player_inputs_map, role_assignments) =
            self.update_strategy_path(&world_data, &team_context, &active_robots);

        // Handle yellow card robot removal
        let allowed_number_of_robots = world_data.current_game_state.max_allowed_bots;
        if active_robots.len() > allowed_number_of_robots as usize {
            let n_robots_to_remove = (active_robots.len() - allowed_number_of_robots as usize)
                .clamp(0, active_robots.len());
            log::info!(
                "We have {} yellow cards and {} robots, removing {} robots",
                world_data.current_game_state.yellow_cards,
                active_robots.len(),
                n_robots_to_remove
            );
            // Sort active robots by id for deterministic removal
            let mut sorted_robots = active_robots.clone();
            sorted_robots.sort();
            self.removing_players
                .extend(sorted_robots.iter().rev().take(n_robots_to_remove));
        }

        // Drive robots that are being removed to the removal position
        let mut player_inputs_map = player_inputs_map;
        let mut removing_players = self.removing_players.iter().copied().collect::<Vec<_>>();
        removing_players.sort();
        let first_removal_position = Vector2::new(
            -800.0 * side_assignment.attacking_direction_sign(team_color), // unflip to world coords
            world_data
                .field_geom
                .as_ref()
                .map(|g| -g.field_width / 2.0 - 150.0)
                .unwrap_or(-3000.0 - 150.0),
        );
        for (i, player_id) in removing_players.iter().enumerate() {
            let player_data = world_data.own_players.iter().find(|p| p.id == *player_id);
            if player_data.is_some() {
                let removal_position =
                    first_removal_position + Vector2::new(-(i as f64) * 100.0, 0.0);
                let mut player_input = PlayerControlInput::default();
                player_input.with_position(removal_position);
                player_inputs_map.insert(*player_id, player_input);
                team_context
                    .player_context(*player_id)
                    .debug_string("role", "removed");
            } else {
                self.removing_players.remove(player_id);
            }
        }

        // Apply role types for compliance and debug display
        for (player_id, input) in player_inputs_map.iter_mut() {
            let player_context = team_context.player_context(*player_id);
            if let Some(role_name) = role_assignments.get(player_id) {
                player_context.debug_string("role", role_name);
                input.role_type = role_type_from_name(role_name);
            } else {
                player_context.debug_string("role", "unassigned");
                input.role_type = RoleType::Player;
            }
        }

        // Prepare inputs for compliance
        let mut inputs_for_comply = PlayerInputs::new();
        for (id, input) in player_inputs_map.iter() {
            self.avoid_goal_area_flags
                .insert(*id, input.role_type != RoleType::Goalkeeper);
            if matches!(
                world_data.current_game_state.game_state,
                GameState::BallReplacement(_)
            ) {
                // In ball placement dont do anything by default
                inputs_for_comply.insert(*id, PlayerControlInput::default());
            } else {
                inputs_for_comply.insert(*id, input.clone());
            }
        }
        let final_player_inputs = if self.comply_enabled {
            comply(&world_data, inputs_for_comply, &team_context)
        } else {
            inputs_for_comply
        };

        let all_players = world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
            .collect::<Vec<_>>();

        // If iLQR mode is active, solve for every robot that has a position
        // target before the per-controller update. Results override the MTP
        // velocity computed inside `controller.update()` below. In MTP mode
        // this map stays empty and the existing MTP path runs untouched.
        let ilqr_overrides: HashMap<PlayerId, Vector2> = match self.settings.controller_mode {
            ControllerMode::Ilqr => {
                let inputs_for_ilqr: HashMap<PlayerId, PlayerControlInput> = self
                    .player_controllers
                    .keys()
                    .map(|id| {
                        let base = final_player_inputs.player(*id);
                        let effective = manual_override.get(id).cloned().unwrap_or(base);
                        (*id, effective)
                    })
                    .collect();
                self.ilqr_controller.compute_batch_control(
                    &self.player_controllers,
                    &inputs_for_ilqr,
                    &world_data,
                    &self.avoid_goal_area_flags,
                )
            }
            ControllerMode::Mtp => HashMap::new(),
        };

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let id = controller.id();
                let player_context = team_context.player_context(id);
                let default_input = final_player_inputs.player(id);
                let input_to_use = manual_override.get(&id).unwrap_or(&default_input);

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                let avoid_ball = input_to_use.avoid_ball;

                let mut obstacles = Vec::new();
                if avoid_ball
                    || world_data.current_game_state.game_state == GameState::PreparePenalty
                    || world_data.current_game_state.game_state == GameState::Stop
                {
                    if let Some(ball) = world_data.ball.as_ref() {
                        obstacles.push(Obstacle::Circle {
                            center: ball.position.xy(),
                            radius: if world_data.current_game_state.game_state == GameState::Stop {
                                800.0
                            } else {
                                100.0 + 100.0 * (input_to_use.care + input_to_use.avoid_ball_care)
                            },
                        });
                    }
                }

                match world_data.current_game_state.game_state {
                    GameState::BallReplacement(target_ball_pos) => {
                        if let Some(ball) = &world_data.ball {
                            obstacles.push(Obstacle::Line {
                                start: ball.position.xy(),
                                end: target_ball_pos,
                            });
                        }
                    }
                    _ => {}
                }

                let avoid_goal_area = if !matches!(
                    world_data.current_game_state.game_state,
                    GameState::BallReplacement(_)
                ) {
                    self.avoid_goal_area_flags.get(&id).copied().unwrap_or(true)
                } else {
                    false
                };
                let avoid_goal_area_margin = match world_data.current_game_state.game_state {
                    GameState::Stop | GameState::FreeKick
                        if input_to_use.role_type != RoleType::Waller =>
                    {
                        500.0
                    }
                    _ => 0.0,
                };

                let dist_to_own_goal = player_data.position.x
                    - world_data
                        .field_geom
                        .as_ref()
                        .map(|f| f.field_length / 2.0)
                        .unwrap_or(0.0);
                let avoid_opp_robots =
                    if input_to_use.role_type == RoleType::Waller && dist_to_own_goal < 1300.0 {
                        false
                    } else {
                        true
                    };

                controller.update(
                    player_data,
                    &world_data,
                    input_to_use,
                    world_data.dt,
                    is_manual,
                    obstacles,
                    &all_players,
                    &player_context,
                    avoid_goal_area,
                    avoid_goal_area_margin,
                    avoid_opp_robots,
                );

                // iLQR (when enabled) overrides the MTP velocity that
                // `update()` just computed. We let `update()` run
                // unconditionally so yaw/kicker/dribbler logic still fires.
                // if let Some(vel) = ilqr_overrides.get(&id) {
                //     controller.set_target_velocity(*vel, world_data.dt);
                //     player_context.debug_string("controller", "iLQR");
                // }
            } else {
                controller.increment_frames_misses();
            }
        }
    }

    /// Update using strategy-controlled path (skill commands).
    fn update_strategy_path(
        &mut self,
        world_data: &Arc<TeamData>,
        team_context: &TeamContext,
        active_robots: &[PlayerId],
    ) -> (
        HashMap<PlayerId, PlayerControlInput>,
        HashMap<PlayerId, String>,
    ) {
        let mut player_inputs = HashMap::new();

        for player_id in active_robots {
            let player_data = match world_data.own_players.iter().find(|p| p.id == *player_id) {
                Some(p) => p,
                None => continue,
            };

            let skill_cmd = self
                .strategy_input
                .skill_commands
                .get(player_id)
                .and_then(|opt| opt.as_ref());

            let ctx = SkillContext {
                player: player_data,
                world: world_data,
                team_context,
                debug_prefix: format!("p{}", player_id),
            };

            let input = self
                .skill_executor
                .process_command(*player_id, skill_cmd, ctx);
            player_inputs.insert(*player_id, input);
        }

        (player_inputs, self.strategy_input.player_roles.clone())
    }

    /// Per-player last computed velocity setpoint in global frame (mm/s).
    /// Used by the test driver to record the controller's actual cmd during
    /// position-controlled motion.
    pub fn target_velocities_global(&self) -> HashMap<PlayerId, Vector2> {
        self.player_controllers
            .iter()
            .map(|(id, c)| (*id, c.target_velocity_global()))
            .collect()
    }

    pub fn commands(
        &mut self,
        side_assignment: SideAssignment,
        color: TeamColor,
    ) -> Vec<PlayerCmd> {
        let untransformer = PlayerCmdUntransformer::new(side_assignment, color);
        let team_context = TeamContext::new(color, side_assignment);
        self.player_controllers
            .values_mut()
            .map(|c| {
                let player_context = team_context.player_context(c.id());
                c.command(&player_context, untransformer.clone())
            })
            .collect()
    }
}

/// Determine role type from role name string.
fn role_type_from_name(role_name: &str) -> RoleType {
    if role_name.contains("goalkeeper") {
        RoleType::Goalkeeper
    } else if role_name.contains("kickoff_kicker") {
        RoleType::KickoffKicker
    } else if role_name.contains("free_kick_kicker") {
        RoleType::FreeKicker
    } else if role_name.contains("waller") {
        RoleType::Waller
    } else if role_name.contains("penalty_kicker") {
        RoleType::PenaltyKicker
    } else {
        RoleType::Player
    }
}

fn comply(world_data: &TeamData, inputs: PlayerInputs, team_context: &TeamContext) -> PlayerInputs {
    if let (Some(ball), Some(field)) = (world_data.ball.as_ref(), world_data.field_geom.as_ref()) {
        let game_state = world_data.current_game_state.game_state;
        let ball_pos = ball.position.xy();

        inputs
            .iter()
            .map(|(id, input)| {
                let player_data = world_data
                    .own_players
                    .iter()
                    .find(|p| p.id == *id)
                    .expect("Player not found in world data");

                let mut new_input = input.clone();

                if matches!(
                    game_state,
                    GameState::Halt | GameState::Unknown | GameState::Timeout
                ) {
                    new_input.with_speed_limit(0.0);
                    new_input.with_angular_speed_limit(0.0);
                    new_input.dribbling_speed = 0.0;
                }

                if matches!(game_state, GameState::Stop) {
                    new_input.with_speed_limit(1500.0);
                }

                let us_operating = world_data.current_game_state.us_operating;
                if matches!(
                    game_state,
                    GameState::PreparePenalty | GameState::Penalty | GameState::PenaltyRun
                ) && ((us_operating && input.role_type != RoleType::PenaltyKicker)
                    || (!us_operating && input.role_type != RoleType::Goalkeeper))
                {
                    let penalty_position_y =
                        -field.field_width / 2.0 + 120.0 + player_data.id.as_u32() as f64 * 220.0;
                    let penalty_positin_x = if us_operating {
                        -field.field_length / 2.0 - 120.0
                    } else {
                        field.field_length / 2.0 + 120.0
                    };
                    new_input.with_position(Vector2::new(penalty_positin_x, penalty_position_y));
                    new_input.with_speed_limit(1500.0);
                    new_input.dribbling_speed = 0.0;
                }
                if matches!(
                    game_state,
                    GameState::PreparePenalty | GameState::Penalty | GameState::PenaltyRun
                ) && input.role_type != RoleType::Goalkeeper
                    && !world_data.current_game_state.us_operating
                {
                    let mut pos = new_input.position.unwrap_or(player_data.position);
                    pos.x = -field.field_length / 2.0;
                    new_input.with_position(pos);
                }

                if matches!(
                    game_state,
                    GameState::Run | GameState::Stop | GameState::FreeKick
                ) {
                    // Avoid goal area
                    let min_distance = match game_state {
                        GameState::Run => 80.0,
                        GameState::Stop | GameState::FreeKick
                            if input.role_type != RoleType::Waller =>
                        {
                            500.0
                        }
                        GameState::Stop | GameState::FreeKick
                            if input.role_type == RoleType::Waller =>
                        {
                            120.0
                        }
                        _ => 80.0,
                    };
                    let max_radius = 4000;
                    let margin = 40.0;
                    // Opponent goal area
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Rectangle {
                            min: Vector2::new(
                                field.field_length / 2.0 - field.penalty_area_depth - margin,
                                -field.penalty_area_width / 2.0 - margin,
                            ),
                            max: Vector2::new(
                                field.field_length / 2.0 + 10_000.0,
                                field.penalty_area_width / 2.0 + margin,
                            ),
                        },
                        min_distance,
                        player_data.position,
                        input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    new_input.with_position(target);

                    // Own goal area
                    if input.role_type != RoleType::Goalkeeper {
                        let max_radius = 4000;
                        let target = dies_core::nearest_safe_pos(
                            dies_core::Avoid::Rectangle {
                                min: Vector2::new(
                                    -field.field_length / 2.0 - 10_000.0,
                                    -field.penalty_area_width / 2.0 - margin,
                                ),
                                max: Vector2::new(
                                    -field.field_length / 2.0 + field.penalty_area_depth + margin,
                                    field.penalty_area_width / 2.0 + margin,
                                ),
                            },
                            20.0,
                            player_data.position,
                            new_input.position.unwrap_or(player_data.position),
                            max_radius,
                            field,
                        );
                        new_input.with_position(target);
                    }
                }

                if game_state == GameState::Stop
                    || (game_state == GameState::FreeKick
                        && input.role_type != RoleType::FreeKicker)
                {
                    if game_state == GameState::Stop {
                        new_input.with_speed_limit(1300.0);
                        new_input.dribbling_speed = 0.0;
                        new_input.kicker = KickerControlInput::Disarm;
                    }

                    let min_distance = 800.0;
                    let max_radius = 4000;
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Circle { center: ball_pos },
                        min_distance,
                        player_data.position,
                        new_input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    new_input.with_position(target);
                }

                if let GameState::BallReplacement(pos) = game_state {
                    let line_start = ball_pos;
                    let line_end = pos;
                    team_context.debug_line_colored(
                        "ball_placement",
                        line_start,
                        line_end,
                        dies_core::DebugColor::Orange,
                    );
                    team_context.debug_cross_colored(
                        "ball_placement_target",
                        pos,
                        dies_core::DebugColor::Orange,
                    );

                    let min_distance = 800.0;
                    let max_radius = 4000;
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Line {
                            start: line_start,
                            end: line_end,
                        },
                        min_distance,
                        player_data.position,
                        new_input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    dies_core::debug_cross(
                        format!("ball_placement_target_{}", id),
                        target,
                        dies_core::DebugColor::Orange,
                    );
                    new_input.with_position(target);
                } else {
                    team_context.debug_remove("ball_placement");
                    team_context.debug_remove("ball_placement_target");
                }

                if matches!(game_state, GameState::PrepareKickoff | GameState::Kickoff)
                    && (input.role_type != RoleType::KickoffKicker
                        || !world_data.current_game_state.us_operating)
                {
                    let mut target_pos = new_input.position.unwrap_or(player_data.position).clone();
                    target_pos.x = target_pos.x.min(-100.0);
                    new_input.with_position(target_pos);
                    // we do it twice to make sure we are on the right side of the field

                    let center_circle_center = Vector2::new(0.0, 0.0);
                    let center_circle_radius = field.center_circle_radius;
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Circle {
                            center: center_circle_center,
                        },
                        center_circle_radius + 500.0,
                        player_data.position,
                        new_input.position.unwrap_or(player_data.position),
                        4000,
                        field,
                    );
                    new_input.with_position(target);

                    // Keep to our side of the field
                    let mut target_pos = new_input.position.unwrap_or(player_data.position).clone();
                    target_pos.x = target_pos.x.min(-100.0);
                    new_input.with_position(target_pos);
                }

                (*id, new_input)
            })
            .collect()
    } else {
        inputs
    }
}
