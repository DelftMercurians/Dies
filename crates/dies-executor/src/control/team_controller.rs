use std::collections::{HashMap, HashSet};

use dies_core::{ExecutorSettings, GameState, PlayerCmd, PlayerId, RoleType, WorldData};

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
};
use crate::{
    roles::{RoleCtx, SkillState},
    strategy::{AdHocStrategy, Strategy, StrategyCtx},
    PlayerControlInput, StrategyMap,
};

const ACTIVATION_TIME: f64 = 0.2;

#[derive(Default)]
struct RoleState {
    skill_map: HashMap<String, SkillState>,
}

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    strategy: StrategyMap,
    active_strat: Option<String>,
    role_states: HashMap<PlayerId, RoleState>,
    settings: ExecutorSettings,
    halt: AdHocStrategy,
    start_time: std::time::Instant,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(strategy: StrategyMap, settings: &ExecutorSettings) -> Self {
        let mut team = Self {
            player_controllers: HashMap::new(),
            strategy,
            role_states: HashMap::new(),
            settings: settings.clone(),
            active_strat: None,
            halt: AdHocStrategy::new(),
            start_time: std::time::Instant::now(),
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

        if self.start_time.elapsed().as_secs_f64() < ACTIVATION_TIME {
            println!("detected_ids: {:?}", detected_ids);
            return;
        }

        let state = world_data.current_game_state.game_state;
        let strategy = if matches!(
            world_data.current_game_state.game_state,
            GameState::BallReplacement(_)
        ) {
            &mut self.halt
        } else if let Some(strategy) = self.strategy.get_strategy(&state) {
            let name = strategy.name().to_owned();
            dies_core::debug_string("active_strat", &name);
            if self.active_strat.as_ref() != Some(&name) {
                log::info!("Switching to strategy: {}", name);
                self.active_strat = Some(name);
                strategy.on_enter(StrategyCtx { world: &world_data });
            }
            strategy
        } else {
            return;
        };

        let strategy_ctx = StrategyCtx { world: &world_data };
        strategy.update(strategy_ctx);

        let mut role_types = HashMap::new();
        let mut inputs =
            world_data
                .own_players
                .iter()
                .fold(PlayerInputs::new(), |mut inputs, player_data| {
                    let id = player_data.id;
                    let strategy_ctx = StrategyCtx { world: &world_data };
                    if let Some(role) = strategy.get_role(id, strategy_ctx) {
                        let role_state = self.role_states.entry(id).or_default();
                        let role_ctx =
                            RoleCtx::new(player_data, &world_data, &mut role_state.skill_map);
                        let mut new_input = role.update(role_ctx);
                        new_input.role_type = role.role_type();
                        role_types.insert(id, new_input.role_type);
                        inputs.insert(id, new_input);
                    } else {
                        inputs.insert(id, PlayerControlInput::new());
                    }
                    inputs
                });

        // If in a stop state, override the inputs
        if matches!(
            world_data.current_game_state.game_state,
            GameState::Stop | GameState::BallReplacement(_) | GameState::FreeKick
        ) {
            inputs = comply(&world_data, inputs);
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

                let is_manual = manual_override
                    .get(&id)
                    .map(|i| !i.velocity.is_zero())
                    .unwrap_or(false);

                let role_type = role_types.get(&id).cloned().unwrap_or_default();
                let obsacles = world_data.get_obstacles_for_player(role_type);

                controller.update(
                    player_data,
                    &world_data,
                    input,
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

    /// Get the currently active commands for the players.
    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values_mut()
            .map(|c| c.command())
            .collect()
    }
}

/// Override the inputs to comply with the stop state.
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
