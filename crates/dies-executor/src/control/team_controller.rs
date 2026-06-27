use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use dies_core::{
    DebugColor, ExecutorSettings, GameState, PlayerCmd, PlayerCmdUntransformer, PlayerId,
    PlayerSkillInfo, RoleType, SideAssignment, TeamColor, TeamData, Vector2, PLAYER_RADIUS,
};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use std::sync::Arc;

use super::{
    avoidance::{AvoidanceGates, GlobalPlanner, ObstacleSet, OrcaSolver},
    joint_skill_executor::JointSkillExecutor,
    pass_coordinator::PassContext,
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    skill_executor::{SkillContext, SkillExecutor},
    team_context::TeamContext,
};
use crate::PlayerControlInput;
use dies_strategy_protocol::PassResult;

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
    /// Joint (multi-robot) skill execution — currently pass coordination.
    joint_skill_executor: JointSkillExecutor,
    /// Current strategy input (skill commands and roles).
    strategy_input: StrategyInput,

    /// Global path planner (owns per-robot path cache for hysteresis).
    planner: GlobalPlanner,
    /// ORCA reciprocal velocity-space avoidance.
    orca: OrcaSolver,

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
            joint_skill_executor: JointSkillExecutor::new(),
            strategy_input: StrategyInput::default(),
            planner: GlobalPlanner::new(&settings.avoidance),
            orca: OrcaSolver::new(&settings.avoidance),
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
        self.planner.update_settings(&settings.avoidance);
        self.orca.update_settings(&settings.avoidance);
        self.settings = settings.clone();
    }

    /// Set strategy input (skill commands and roles from strategy host).
    pub fn set_strategy_input(&mut self, input: StrategyInput) {
        self.strategy_input = input;
    }

    /// Get current skill statuses for all players.
    ///
    /// Joint (pass) statuses override per-player skill statuses for robots
    /// currently or recently in a pass.
    pub fn get_skill_statuses(&self) -> HashMap<PlayerId, SkillStatus> {
        let mut statuses = self.skill_executor.get_all_statuses();
        for (id, status) in self.joint_skill_executor.statuses() {
            statuses.insert(*id, *status);
        }
        statuses
    }

    /// Get rich, UI-facing skill info for all players.
    ///
    /// Joint (pass) infos override per-player skill infos for robots currently or
    /// recently in a pass, mirroring [`Self::get_skill_statuses`].
    pub fn get_skill_infos(&self) -> HashMap<PlayerId, PlayerSkillInfo> {
        let mut infos = self.skill_executor.get_all_infos();
        for (id, info) in self.joint_skill_executor.get_all_infos() {
            infos.insert(id, info);
        }
        infos
    }

    /// Get rich pass results for players involved in a pass (empty otherwise).
    pub fn get_pass_results(&self) -> HashMap<PlayerId, PassResult> {
        self.joint_skill_executor.pass_results().clone()
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
                        input_to_use.position.map(|p| vec![p]),
                        world_data.dt,
                        false,
                        &player_context,
                        true,
                        &[],
                        &[],
                        None,
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
        let goal_area_avoidance_enabled = self.settings.goal_area_avoidance;
        let mut inputs_for_comply = PlayerInputs::new();
        for (id, input) in player_inputs_map.iter() {
            self.avoid_goal_area_flags.insert(
                *id,
                goal_area_avoidance_enabled && input.role_type != RoleType::Goalkeeper,
            );
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
            comply(
                &world_data,
                inputs_for_comply,
                &team_context,
                goal_area_avoidance_enabled,
            )
        } else {
            inputs_for_comply
        };

        let cfg = self.settings.avoidance.clone();
        let game_state = world_data.current_game_state.game_state;
        let dt = world_data.dt.clamp(1.0e-3, 0.5);

        // Drop cached planner paths for robots we no longer control.
        let live_ids: HashSet<PlayerId> = self.player_controllers.keys().copied().collect();
        self.planner.retain(&live_ids);

        // Debug (team-scoped): keep-out radii around each robot. The inner circle
        // is ORCA's hard combined radius (2·PLAYER_RADIUS + clearance) — within
        // it, robot centres collide. The outer circle is what the planner routes
        // around (+ planner_margin), kept wider so it sits outside ORCA's braking
        // band. Own robots blue, opponents red; planner radius in orange.
        let orca_keepout = 2.0 * PLAYER_RADIUS + cfg.robot_clearance;
        let planner_keepout = orca_keepout + cfg.planner_margin;
        for p in world_data.own_players.iter() {
            team_context.debug_circle_stroke_colored(
                format!("avoid.keepout.own{}", p.id),
                p.position,
                orca_keepout,
                DebugColor::Blue,
            );
            team_context.debug_circle_stroke_colored(
                format!("avoid.plankeepout.own{}", p.id),
                p.position,
                planner_keepout,
                DebugColor::Orange,
            );
        }
        for p in world_data.opp_players.iter() {
            team_context.debug_circle_stroke_colored(
                format!("avoid.keepout.opp{}", p.id),
                p.position,
                orca_keepout,
                DebugColor::Red,
            );
            team_context.debug_circle_stroke_colored(
                format!("avoid.plankeepout.opp{}", p.id),
                p.position,
                planner_keepout,
                DebugColor::Orange,
            );
        }

        // Per robot: build the obstacle set, plan a path, then run the controller
        // — which follows the path, deflects the result through ORCA against the
        // obstacle set's agents, and acceleration-clamps the command, all in one
        // place. ORCA is gated per-robot below; each per-controller solve is
        // independent (reciprocity works through neighbours' observed velocities).
        for controller in self.player_controllers.values_mut() {
            let id = controller.id();
            let Some(player_data) = world_data.own_players.iter().find(|p| p.id == id) else {
                controller.increment_frames_misses();
                continue;
            };
            let player_context = team_context.player_context(id);

            let effective_input = manual_override
                .get(&id)
                .cloned()
                .unwrap_or_else(|| final_player_inputs.player(id));
            let is_manual = manual_override
                .get(&id)
                .map(|i| !i.velocity.is_zero())
                .unwrap_or(false);

            // Goal-area avoidance is off during ball placement and for the
            // goalkeeper (via the per-robot flag).
            let avoid_goal_area = if !matches!(game_state, GameState::BallReplacement(_)) {
                self.avoid_goal_area_flags.get(&id).copied().unwrap_or(true)
            } else {
                false
            };

            // A waller hugging our own goal is allowed to ignore opponents so it
            // can hold the line.
            let dist_to_own_goal = player_data.position.x
                - world_data
                    .field_geom
                    .as_ref()
                    .map(|f| f.field_length / 2.0)
                    .unwrap_or(0.0);
            let avoid_opp_robots =
                !(effective_input.role_type == RoleType::Waller && dist_to_own_goal < 1300.0);

            let gates = AvoidanceGates {
                avoid_defense_area: avoid_goal_area,
                avoid_ball: effective_input.avoid_ball,
                avoid_ball_care: effective_input.avoid_ball_care,
                avoid_opp_robots,
                is_kickoff_kicker: effective_input.role_type == RoleType::KickoffKicker,
            };
            let obstacles = ObstacleSet::build(&world_data, id, gates, game_state, &cfg);

            // ORCA gated per-robot: a robot with `avoid_robots = false` (e.g. the
            // goalkeeper) has its commanded velocity passed through untouched —
            // no reciprocal avoidance and, crucially, no static field-boundary
            // wall deflection.
            let orca = if cfg.orca_enabled && effective_input.avoid_robots {
                Some(&self.orca)
            } else {
                None
            };

            // Global planner → full path to follow (LOS fast path inside).
            // Draws the route under the player's `plan.*` keys. Gated per-robot:
            // `use_planner = false` drives straight at the target.
            let path = effective_input.position.map(|target| {
                if cfg.planner_enabled && effective_input.use_planner {
                    self.planner.plan(
                        id,
                        player_data.position,
                        target,
                        PLAYER_RADIUS,
                        &obstacles,
                        &player_context,
                    )
                } else {
                    vec![target]
                }
            });

            controller.update(
                player_data,
                &world_data,
                &effective_input,
                path,
                dt,
                is_manual,
                &player_context,
                avoid_goal_area,
                &obstacles.agents,
                &obstacles.statics,
                orca,
            );
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

        // Joint (pass) coordination runs first, as a single first-class tick. It
        // claims a set of robots (`managed`); those are skipped below so the
        // per-player skill executor never sees a `Pass` command.
        let pass_ctx = PassContext {
            world: world_data,
            team_context,
        };
        let joint = self
            .joint_skill_executor
            .tick_all(&self.strategy_input.skill_commands, &pass_ctx);

        for player_id in active_robots {
            let player_data = match world_data.own_players.iter().find(|p| p.id == *player_id) {
                Some(p) => p,
                None => continue,
            };

            if joint.managed.contains(player_id) {
                let input = joint.inputs.get(player_id).cloned().unwrap_or_default();
                player_inputs.insert(*player_id, input);
                continue;
            }

            let skill_cmd = self
                .strategy_input
                .skill_commands
                .get(player_id)
                .and_then(|opt| opt.as_ref());

            let ctx = SkillContext {
                player: player_data,
                world: world_data,
                team_context,
                debug_prefix: team_context.key(format!("p{}", player_id)),
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

    /// Players that will kick on this tick (efference copy for possession).
    pub fn kicking_players(&self) -> Vec<PlayerId> {
        self.player_controllers
            .values()
            .filter(|c| c.is_kicking())
            .map(|c| c.id())
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

fn comply(
    world_data: &TeamData,
    inputs: PlayerInputs,
    team_context: &TeamContext,
    goal_area_avoidance_enabled: bool,
) -> PlayerInputs {
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

                if goal_area_avoidance_enabled
                    && matches!(
                        game_state,
                        GameState::Run | GameState::Stop | GameState::FreeKick
                    )
                {
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
                    team_context.debug_cross_colored(
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
