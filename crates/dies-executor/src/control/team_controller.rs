use std::{
    collections::{HashMap, HashSet},
    time::{Duration, Instant},
};

use dies_core::{
    ExecutorSettings, GameState, Obstacle, PlayerCmd, PlayerCmdUntransformer, PlayerId, RoleType,
    SideAssignment, TeamColor, TeamData, Vector2
};
use std::sync::Arc;

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    team_context::TeamContext,
};
use crate::{
    behavior_tree::{
        BehaviorTree, BtContext, GameContext, RobotSituation, RoleAssignmentSolver, Strategy,
    },
    PlayerControlInput,
};

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
    role_solver: RoleAssignmentSolver,
    role_assignments: Arc<HashMap<PlayerId, String>>,

    // mpc stuff
    start_time: std::time::Instant,
    #[cfg(feature = "mpc")]
    mpc_controller: MPCController,

    // bht stuff
    strategy: Strategy,
    player_behavior_trees: HashMap<PlayerId, BehaviorTree>,
    bt_context: BtContext,
    team_color: TeamColor,

    // last_keeper_id: Option<PlayerId>,
    removing_players: HashSet<PlayerId>,

    warmup_timer: Option<Instant>,
    warmup_done: bool,

    // per-robot flags
    avoid_goal_area_flags: HashMap<PlayerId, bool>,
}

impl TeamController {
    pub fn new(settings: &ExecutorSettings, team_color: TeamColor, strategy: Strategy) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
            role_solver: RoleAssignmentSolver::new(),
            role_assignments: Arc::new(HashMap::new()),
            start_time: std::time::Instant::now(),
            #[cfg(feature = "mpc")]
            mpc_controller: MPCController::new(),

            // bht stuff
            strategy,
            player_behavior_trees: HashMap::new(),
            bt_context: BtContext::new(),
            team_color,

            removing_players: HashSet::new(),

            // per-robot flags - initialize to true by default
            avoid_goal_area_flags: HashMap::new(),

            warmup_timer: None,
            warmup_done: false,
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

        let mut player_inputs_map: HashMap<PlayerId, PlayerControlInput> = HashMap::new();

        // Get active robots
        let active_robots: Vec<PlayerId> = world_data
            .own_players
            .iter()
            .map(|p| p.id)
            .filter(|id| !self.removing_players.contains(id))
            .collect();

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

            // Update player controllers
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
                    );
                }
            }

            return;
        }

        // Get role assignment problem from script
        let mut game_context = GameContext::new(Arc::clone(&world_data));
        (self.strategy)(&mut game_context);
        let assignment_problem = game_context.into_role_assignment_problem();

        // Solve role assignments
        let priority_list = match self.role_solver.solve(
            &assignment_problem,
            &active_robots,
            team_context.clone(),
            world_data.clone(),
            Some(&self.role_assignments),
        ) {
            Ok((assignments, priority_list)) => {
                self.role_assignments = Arc::new(assignments.clone());

                let priority_list_ids: Vec<PlayerId> = priority_list
                    .iter()
                    .filter_map(|role_name| {
                        assignments
                            .iter()
                            .find(|(_, role)| *role == role_name)
                            .map(|(id, _)| *id)
                    })
                    .collect();

                // Update behavior trees based on new assignments
                for player_id in &active_robots {
                    let role_name = assignments
                        .iter()
                        .find(|(id, _)| *id == player_id)
                        .map(|(_, role)| role.clone())
                        .unwrap_or_default();
                    let player_context = team_context.player_context(*player_id);

                    // Only rebuild tree if role changed
                    let needs_rebuild = self
                        .player_behavior_trees
                        .get(player_id)
                        .map(|bt| bt.name != role_name)
                        .unwrap_or(true);

                    if needs_rebuild {
                        // println!("assigned {} to {} with rebuild", role_name, player_id);
                        // Clear semaphores for this player before rebuilding
                        self.bt_context.clear_semaphores_for_player(*player_id);

                        // Find the role and build its tree
                        if let Some(role) = assignment_problem
                            .roles
                            .iter()
                            .find(|r| r.name == role_name)
                        {
                            let situation = RobotSituation::new(
                                *player_id,
                                world_data.clone(),
                                self.bt_context.clone(),
                                player_context.key("bt"),
                                self.role_assignments.clone(),
                                team_color,
                            );

                            self.player_behavior_trees.insert(
                                *player_id,
                                BehaviorTree::new(
                                    role_name.clone(),
                                    (role.tree_builder)(&situation),
                                ),
                            );
                        } else {
                            self.player_behavior_trees.remove(player_id);
                        }
                    }
                }
                priority_list_ids
            }
            Err(e) => {
                log::error!("Failed to solve role assignments: {}", e);
                // Fallback to sorting by id
                let mut priority_list_ids: Vec<PlayerId> = active_robots.clone();
                priority_list_ids.sort();
                priority_list_ids
            }
        };

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
            self.removing_players
                .extend(priority_list.iter().rev().take(n_robots_to_remove));
        }

        // Clean up empty semaphores periodically
        self.bt_context.cleanup_empty_semaphores();

        // Execute behavior trees
        for player_data in &world_data.own_players {
            let player_context = team_context.player_context(player_data.id);
            if self.removing_players.contains(&player_data.id) {
                player_context.debug_string("role", "removed");
                continue;
            }

            let player_id = player_data.id;
            let player_bt = self.player_behavior_trees.entry(player_id).or_default();

            let viz_path_prefix = team_context.player_context(player_id).key("bt");
            let mut robot_situation = RobotSituation::new(
                player_id,
                world_data.clone(),
                self.bt_context.clone(),
                viz_path_prefix,
                self.role_assignments.clone(),
                team_color,
            );

            player_context.debug_string("role_bt", &player_bt.name);
            let (_status, player_input_opt) = player_bt.tick(&mut robot_situation);
            let mut player_input = player_input_opt.unwrap_or_else(PlayerControlInput::default);

            // Set role type based on assignment
            if let Some(role_name) = self.role_assignments.get(&player_id) {
                player_context.debug_string("role", role_name);
                player_input.role_type = if role_name.contains("goalkeeper") {
                    RoleType::Goalkeeper
                } else if role_name.contains("kickoff_kicker") {
                    RoleType::KickoffKicker
                } else if role_name.contains("free_kick_kicker") {
                    RoleType::FreeKicker
                } else if role_name.contains("waller") {
                    RoleType::Waller
                } else {
                    RoleType::Player
                };
            } else {
                player_context.debug_string("role", "not found");
                player_input.role_type = RoleType::Player;
            }

            player_inputs_map.insert(player_id, player_input);
        }

        // Drive robots that are being removed to the removal position
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
            } else {
                self.removing_players.remove(player_id);
            }
        }

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
        let final_player_inputs = comply(&world_data, inputs_for_comply, &team_context);

        let all_players = world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
            .collect::<Vec<_>>();

        // Collect robots that need MPC processing
        #[allow(unused_mut)]
        let mut mpc_controls: HashMap<PlayerId, Vector2> = HashMap::new();
        #[cfg(feature = "mpc")]
        {
            let mut mpc_robots = Vec::new();
            let mut controllable_mask = Vec::new();
            for controller in self.player_controllers.values() {
                if controller.use_mpc() {
                    let player_data = world_data
                        .own_players
                        .iter()
                        .find(|p| p.id == controller.id());

                    if let Some(player_data) = player_data {
                        let id = controller.id();
                        let default_input = final_player_inputs.player(id);
                        let input = manual_override.get(&id).unwrap_or(&default_input);

                        if let Some(target_pos) = input.position {
                            #[cfg(feature = "mpc")]
                            mpc_robots.push(super::mpc::RobotState {
                                id: controller.id(),
                                position: player_data.position,
                                velocity: player_data.velocity,
                                target_position: target_pos,
                                vel_limit: controller.get_max_speed(),
                            });

                            controllable_mask.push(true); // Robot should be controlled
                        } else {
                            // if we don't need to move - force robots to stay in place while
                            // still putting them to mpc for continuity, which is fairly important
                            #[cfg(feature = "mpc")]
                            mpc_robots.push(super::mpc::RobotState {
                                id: controller.id(),
                                position: player_data.position,
                                velocity: player_data.velocity,
                                target_position: player_data.position,
                                vel_limit: 0.0,
                            });
                            controllable_mask.push(false); // Robot should not be controlled
                        }
                    }
                }
            }

            // Compute batched MPC controls
            #[cfg(feature = "mpc")]
            self.mpc_controller.set_field_bounds(&world_data);

            #[cfg(feature = "mpc")]
            if !mpc_robots.is_empty() {
                // Collect avoid_goal_area flags for MPC robots (in the same order as mpc_robots)
                let mut avoid_goal_area_flags = Vec::new();
                for robot in &mpc_robots {
                    let avoid_goal_area = self
                        .avoid_goal_area_flags
                        .get(&robot.id)
                        .copied()
                        .unwrap_or(true);
                    avoid_goal_area_flags.push(avoid_goal_area);
                }

                mpc_controls = self.mpc_controller.compute_batch_control(
                    &mpc_robots,
                    &world_data,
                    Some(&controllable_mask),
                    &avoid_goal_area_flags,
                )
            }
        }
        // Update the player controllers
        // let trajectories = self.mpc_controller.get_trajectories();
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

                // Handle MPC vs MTP control
                if controller.use_mpc() {
                    if let Some(mpc_control) = mpc_controls.get(&id) {
                        // Use MPC control (override MTP fallback)
                        controller.set_target_velocity(*mpc_control);
                        // Update debug string to show MPC is being used
                        player_context.debug_string("controller", "MPC");
                        // Debug output for MPC timing
                        #[cfg(feature = "mpc")]
                        dies_core::debug_value(
                            format!("p{}.mpc.duration_ms", id),
                            self.mpc_controller.last_solve_time_ms(),
                        );

                        // Plot MPC trajectory if available
                        // if let Some(trajectory) = trajectories.get(&id) {
                        //     let debug_name = format!("mpc_traj_p{}", id);
                        // Clear previous trajectory
                        // dies_core::debug_remove(&debug_name);
                        // Plot trajectory as connected line segments
                        // for i in 0..trajectory.len().saturating_sub(1) {
                        //     if trajectory[i].len() >= 5 && trajectory[i + 1].len() >= 5 {
                        //         let start =
                        //             dies_core::Vector2::new(trajectory[i][1], trajectory[i][2]);
                        //         let end = dies_core::Vector2::new(
                        //             trajectory[i + 1][1],
                        //             trajectory[i + 1][2],
                        //         );
                        //         dies_core::debug_line(
                        //             &format!("{}_seg{}", debug_name, i),
                        //             start,
                        //             end,
                        //             dies_core::DebugColor::Purple,
                        //         );
                        //     }
                        // }
                        // Mark trajectory endpoints
                        // if !trajectory.is_empty() {
                        //     if trajectory[0].len() >= 5 {
                        //         let start_pos =
                        //             dies_core::Vector2::new(trajectory[0][1], trajectory[0][2]);
                        //         dies_core::debug_cross(
                        //             &format!("{}_start", debug_name),
                        //             start_pos,
                        //             dies_core::DebugColor::Green,
                        //         );
                        //     }
                        //     if let Some(last) = trajectory.last() {
                        //         if last.len() >= 5 {
                        //             let end_pos = dies_core::Vector2::new(last[1], last[2]);
                        //             dies_core::debug_circle_fill(
                        //                 &format!("{}_end", debug_name),
                        //                 end_pos,
                        //                 80.0,
                        //                 dies_core::DebugColor::Purple,
                        //             );
                        //         }
                        //     }
                        // }
                        // }
                    }
                    // If MPC returned empty (fallback to MTP), don't call set_target_velocity
                    // Let the controller use MTP through the regular update() call below
                }

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                let avoid_ball = input_to_use.avoid_ball;

                let mut obstacles = Vec::new();
                if avoid_ball {
                    if let Some(ball) = world_data.ball.as_ref() {
                        obstacles.push(Obstacle::Circle {
                            center: ball.position.xy(),
                            radius: 100.0 + 100.0 * input_to_use.care,
                        });
                    }
                }

                match world_data.current_game_state.game_state {
                    GameState::BallReplacement(target_ball_pos) => {
                        if let Some(ball) = &world_data.ball {
                            obstacles.push(Obstacle::Line {
                                start: ball.position.xy(),
                                end: target_ball_pos
                            });
                        }
                    }
                    _ => {}
                }

                // Get the avoid_goal_area flag for this specific player (default to true)
                let avoid_goal_area = if !matches!(
                    world_data.current_game_state.game_state,
                    GameState::BallReplacement(_)
                ) {
                    self.avoid_goal_area_flags.get(&id).copied().unwrap_or(true)
                } else {
                    false
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
                );
            } else {
                controller.increment_frames_misses();
            }
        }
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

                if matches!(game_state, GameState::Halt | GameState::Unknown) {
                    new_input.with_speed_limit(0.0);
                    new_input.with_angular_speed_limit(0.0);
                    new_input.dribbling_speed = 0.0;
                }

                if matches!(game_state, GameState::Stop) {
                    new_input.with_speed_limit(1500.0);
                }

                if matches!(
                    game_state,
                    GameState::Run | GameState::Stop | GameState::FreeKick
                ) {
                    // Avoid goal area
                    let min_distance = match game_state {
                        GameState::Run => 80.0,
                        GameState::Stop | GameState::FreeKick => 200.0,
                        _ => unreachable!(),
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
