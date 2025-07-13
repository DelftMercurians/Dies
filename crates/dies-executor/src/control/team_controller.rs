use std::collections::{HashMap, HashSet};

use dies_core::{
    ExecutorSettings, GameState, PlayerCmd, PlayerCmdUntransformer, PlayerId, RoleType,
    SideAssignment, TeamColor, TeamData,
};
use std::sync::Arc;

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
    team_context::TeamContext,
    MPCController, RobotState,
};
use crate::{
    behavior_tree::{
        BehaviorTree, BtContext, GameContext, RobotSituation, RoleAssignmentSolver, Strategy,
    },
    PlayerControlInput, ScriptError,
};

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
    role_solver: RoleAssignmentSolver,
    role_assignments: HashMap<PlayerId, String>,

    // mpc stuff
    start_time: std::time::Instant,
    mpc_controller: MPCController,

    // bht stuff
    strategy: Strategy,
    player_behavior_trees: HashMap<PlayerId, BehaviorTree>,
    bt_context: BtContext,
    team_color: TeamColor,
    latest_script_errors: Vec<ScriptError>,

    // per-robot flags
    avoid_goal_area_flags: HashMap<PlayerId, bool>,
}

impl TeamController {
    pub fn new(settings: &ExecutorSettings, team_color: TeamColor, strategy: Strategy) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
            role_solver: RoleAssignmentSolver::new(),
            role_assignments: HashMap::new(),
            start_time: std::time::Instant::now(),
            mpc_controller: MPCController::new(),

            // bht stuff
            strategy,
            player_behavior_trees: HashMap::new(),
            bt_context: BtContext::new(),
            team_color,
            latest_script_errors: Vec::new(),

            // per-robot flags - initialize to true by default
            avoid_goal_area_flags: HashMap::new(),
        };
        team.update_controller_settings(settings);
        team
    }

    /// Get and clear any script errors that occurred since the last call
    pub fn take_script_errors(&mut self) -> Vec<ScriptError> {
        std::mem::take(&mut self.latest_script_errors)
    }

    pub fn update_controller_settings(&mut self, settings: &ExecutorSettings) {
        for controller in self.player_controllers.values_mut() {
            controller.update_settings(&settings.controller_settings);
        }
        self.settings = settings.clone();
    }

    /// Enable two-step MTP controller for a specific player
    pub fn set_player_two_step_mtp(&mut self, player_id: PlayerId, enable: bool) {
        if let Some(controller) = self.player_controllers.get_mut(&player_id) {
            controller.set_use_two_step_mtp(enable);
        }
    }

    /// Enable two-step MTP controller for all players
    pub fn set_all_players_two_step_mtp(&mut self, enable: bool) {
        for controller in self.player_controllers.values_mut() {
            controller.set_use_two_step_mtp(enable);
        }
    }

    /// Set avoid_goal_area flag for a specific player
    pub fn set_player_avoid_goal_area(&mut self, player_id: PlayerId, avoid: bool) {
        self.avoid_goal_area_flags.insert(player_id, avoid);
    }

    /// Set avoid_goal_area flag for all players
    pub fn set_all_players_avoid_goal_area(&mut self, avoid: bool) {
        for &player_id in self.player_controllers.keys() {
            self.avoid_goal_area_flags.insert(player_id, avoid);
        }
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
        let active_robots: Vec<PlayerId> = world_data.own_players.iter().map(|p| p.id).collect();

        // Get role assignment problem from script
        let mut game_context = GameContext::new(&world_data);
        (self.strategy)(&mut game_context);
        let assignment_problem = game_context.into_role_assignment_problem();

        // Solve role assignments
        match self.role_solver.solve(
            &assignment_problem,
            &active_robots,
            team_context.clone(),
            world_data.clone(),
            Some(&self.role_assignments),
        ) {
            Ok(assignments) => {
                self.role_assignments = assignments.clone();

                // Update behavior trees based on new assignments
                for (player_id, role_name) in &assignments {
                    let player_context = team_context.player_context(*player_id);
                    player_context.debug_string("role", role_name);

                    // Only rebuild tree if role changed
                    let needs_rebuild = self
                        .player_behavior_trees
                        .get(player_id)
                        .map(|bt| bt.name != *role_name)
                        .unwrap_or(true);

                    if needs_rebuild {
                        // Find the role and build its tree
                        if let Some(role) = assignment_problem
                            .roles
                            .iter()
                            .find(|r| &r.name == role_name)
                        {
                            let situation = RobotSituation::new(
                                *player_id,
                                world_data.clone(),
                                self.bt_context.clone(),
                                player_context.key("bt"),
                            );

                            self.player_behavior_trees.insert(
                                *player_id,
                                BehaviorTree::new(
                                    role_name.clone(),
                                    (role.tree_builder)(&situation),
                                ),
                            );
                        } else {
                            log::error!("Role '{}' not found for player {}", role_name, player_id);
                        }
                    }
                }
            }
            Err(e) => log::error!("Failed to solve role assignments: {}", e),
        }

        // Execute behavior trees
        for player_data in &world_data.own_players {
            let player_id = player_data.id;
            let player_bt = self.player_behavior_trees.entry(player_id).or_default();

            let viz_path_prefix = team_context.player_context(player_id).key("bt");
            let mut robot_situation = RobotSituation::new(
                player_id,
                world_data.clone(),
                self.bt_context.clone(),
                viz_path_prefix,
            );

            let (_status, player_input_opt) = player_bt.tick(&mut robot_situation);
            let mut player_input = player_input_opt.unwrap_or_else(PlayerControlInput::default);

            // Set role type based on assignment
            if let Some(role_name) = self.role_assignments.get(&player_id) {
                player_input.role_type = match role_name.as_str() {
                    "goalkeeper" => RoleType::Goalkeeper,
                    "kickoff_kicker" => RoleType::KickoffKicker,
                    "free_kick_kicker" => RoleType::FreeKicker,
                    _ => RoleType::Player,
                };
            } else if player_id == PlayerId::new(0) {
                player_input.role_type = RoleType::Goalkeeper;
            } else {
                player_input.role_type = RoleType::Player;
            }

            player_inputs_map.insert(player_id, player_input);
        }


        let mut inputs_for_comply = PlayerInputs::new();
        for (id, input) in player_inputs_map.iter() {
            self.avoid_goal_area_flags.insert(*id, input.role_type != RoleType::Goalkeeper);
            inputs_for_comply.insert(*id, input.clone());
        }

        let final_player_inputs = if matches!(
            world_data.current_game_state.game_state,
            GameState::Stop | GameState::BallReplacement(_) | GameState::FreeKick
        ) {
            // comply(&world_data, inputs_for_comply, &team_context)
            inputs_for_comply
        } else {
            inputs_for_comply
        };

        let all_players = world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
            .collect::<Vec<_>>();

        // Collect robots that need MPC processing
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
                        mpc_robots.push(RobotState {
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
                        mpc_robots.push(RobotState {
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
        self.mpc_controller.set_field_bounds(&world_data);
        let mpc_controls = if !mpc_robots.is_empty() {
            // Collect avoid_goal_area flags for MPC robots (in the same order as mpc_robots)
            let mut avoid_goal_area_flags = Vec::new();
            for robot in &mpc_robots {
                let avoid_goal_area = self.avoid_goal_area_flags.get(&robot.id).copied().unwrap_or(true);
                avoid_goal_area_flags.push(avoid_goal_area);
            }

            self.mpc_controller.compute_batch_control(
                &mpc_robots,
                &world_data,
                Some(&controllable_mask),
                &avoid_goal_area_flags
            )
        } else {
            HashMap::new()
        };

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

                let role_type = input_to_use.role_type;
                let obstacles = world_data.get_obstacles_for_player(role_type);

                // Get the avoid_goal_area flag for this specific player (default to true)
                let avoid_goal_area = self.avoid_goal_area_flags.get(&id).copied().unwrap_or(true);

                controller.update(
                    player_data,
                    &world_data,
                    input_to_use,
                    world_data.dt,
                    is_manual,
                    obstacles,
                    &all_players,
                    &player_context,
                    avoid_goal_area
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
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Circle { center: ball_pos },
                        min_distance,
                        player_data.position,
                        input.position.unwrap_or(player_data.position),
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
                        input.position.unwrap_or(player_data.position),
                        max_radius,
                        field,
                    );
                    new_input.with_position(target);
                } else {
                    team_context.debug_remove("ball_placement");
                    team_context.debug_remove("ball_placement_target");
                }

                (*id, new_input)
            })
            .collect()
    } else {
        inputs
    }
}
