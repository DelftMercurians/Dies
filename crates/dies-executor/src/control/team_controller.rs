use std::collections::{HashMap, HashSet};

use dies_core::{ExecutorSettings, GameState, PlayerCmd, PlayerId, RoleType, WorldData};
use std::sync::Arc;

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
};
use crate::{
    behavior_tree::{BehaviorTree, RhaiHost, RobotSituation, TeamContext},
    PlayerControlInput,
};

const ACTIVATION_TIME: f64 = 0.2;

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
    start_time: std::time::Instant,
    player_behavior_trees: HashMap<PlayerId, BehaviorTree>,
    team_context: TeamContext,
    script_host: RhaiHost,
}

impl TeamController {
    pub fn new(settings: &ExecutorSettings) -> Self {
        let main_bt_script_path = "crates/dies-executor/src/bt_scripts/standard_player_tree.rhai";
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
            start_time: std::time::Instant::now(),
            player_behavior_trees: HashMap::new(),
            team_context: TeamContext::new(),
            script_host: RhaiHost::new(main_bt_script_path),
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

    pub fn update(
        &mut self,
        world_data: WorldData,
        manual_override: HashMap<PlayerId, PlayerControlInput>,
    ) {
        let world_data = Arc::new(world_data);
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

        if self.start_time.elapsed().as_secs_f64() < ACTIVATION_TIME {
            return;
        }

        self.team_context.clear_semaphores();

        let mut player_inputs_map: HashMap<PlayerId, PlayerControlInput> = HashMap::new();

        let engine = self.script_host.engine();
        let engine_guard = engine.read().unwrap();
        for player_data in &world_data.own_players {
            let player_id = player_data.id;

            let player_bt =
                self.player_behavior_trees
                    .entry(player_id)
                    .or_insert_with(|| match self.script_host.build_player_bt(player_id) {
                        Ok(bt) => bt,
                        Err(e) => {
                            log::error!("Failed to build player BT: {:?}", e);
                            BehaviorTree::default()
                        }
                    });

            let viz_path_prefix = format!("p{}", player_id);
            let mut robot_situation = RobotSituation::new(
                player_id,
                world_data.clone(),
                self.team_context.clone(),
                viz_path_prefix,
            );

            let (_status, player_input_opt) = player_bt.tick(&mut robot_situation, &engine_guard);
            let mut player_input = player_input_opt.unwrap_or_else(PlayerControlInput::default);

            if player_input.role_type == RoleType::Player {
                if player_id == PlayerId::new(0) {
                    player_input.role_type = RoleType::Goalkeeper;
                } else {
                    player_input.role_type = RoleType::Player;
                }
            }

            player_inputs_map.insert(player_id, player_input);
        }

        let mut inputs_for_comply = PlayerInputs::new();
        for (id, input) in player_inputs_map.iter() {
            inputs_for_comply.insert(*id, input.clone());
        }

        let final_player_inputs = if matches!(
            world_data.current_game_state.game_state,
            GameState::Stop | GameState::BallReplacement(_) | GameState::FreeKick
        ) {
            comply(&world_data, inputs_for_comply)
        } else {
            inputs_for_comply
        };

        let all_players = world_data
            .own_players
            .iter()
            .chain(world_data.opp_players.iter())
            .collect::<Vec<_>>();

        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let id = controller.id();
                let default_input = final_player_inputs.player(id);
                let input_to_use = manual_override.get(&id).unwrap_or(&default_input);

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                let role_type = input_to_use.role_type;
                let obsacles = world_data.get_obstacles_for_player(role_type);

                controller.update(
                    player_data,
                    &world_data,
                    input_to_use,
                    world_data.dt,
                    is_manual,
                    obsacles,
                    &all_players,
                );
            } else {
                controller.increment_frames_misses();
            }
        }
    }

    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values_mut()
            .map(|c| c.command())
            .collect()
    }
}

fn comply(world_data: &WorldData, inputs: PlayerInputs) -> PlayerInputs {
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
