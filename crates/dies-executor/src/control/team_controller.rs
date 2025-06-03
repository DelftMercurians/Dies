use std::collections::{HashMap, HashSet};

use dies_core::{ExecutorSettings, GameState, PlayerCmd, PlayerId, RoleType, WorldData};

use super::{
    player_controller::PlayerController,
    player_input::{KickerControlInput, PlayerInputs},
};
use crate::{
    behavior_tree::{BehaviorNode, RobotSituation, TeamContext},
    PlayerControlInput,
};

const ACTIVATION_TIME: f64 = 0.2;

pub struct TeamController {
    player_controllers: HashMap<PlayerId, PlayerController>,
    settings: ExecutorSettings,
    start_time: std::time::Instant,
    player_behavior_trees: HashMap<PlayerId, Box<dyn BehaviorNode>>,
    team_context: TeamContext,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(settings: &ExecutorSettings) -> Self {
        let team_context = TeamContext::new();
        let mut team = Self {
            player_controllers: HashMap::new(),
            settings: settings.clone(),
            start_time: std::time::Instant::now(),
            player_behavior_trees: HashMap::new(),
            team_context,
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

        // Reset team context (e.g., clear semaphores)
        self.team_context.clear_semaphores();

        let mut player_inputs_map: HashMap<PlayerId, PlayerControlInput> = HashMap::new();

        for player_data in &world_data.own_players {
            let player_id = player_data.id;

            // Get or build the behavior tree for the player
            // For now, let's use a placeholder builder if a BT doesn't exist.
            // In a real scenario, BTs would be built/assigned based on game state, player roles, etc.
            let player_bt = self
                .player_behavior_trees
                .entry(player_id)
                .or_insert_with(|| {
                    build_player_bt(player_id, &world_data, &self.settings) // Placeholder
                });

            // Create RobotSituation
            // The viz_path_prefix should be meaningful, e.g., "player_X_bt"
            let viz_path_prefix = format!("p{}", player_id);
            let mut robot_situation =
                RobotSituation::new(player_id, &world_data, &self.team_context, viz_path_prefix);

            // Tick the behavior tree
            let (_status, player_input_opt) = player_bt.tick(&mut robot_situation);
            let mut player_input = player_input_opt.unwrap_or_else(PlayerControlInput::default);

            // Ensure role_type is set, if not already by BT.
            // This is important for `comply` and `get_obstacles_for_player`.
            // A more sophisticated BT would set this.
            if player_input.role_type == RoleType::Player {
                // Default to Player, or decide based on player_id (e.g. goalie)
                if player_id == PlayerId::new(0) {
                    // Assuming goalie might be ID 0
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

        // If in a stop state, override the inputs
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

        // Update the player controllers
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

    /// Get the currently active commands for the players.
    pub fn commands(&mut self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values_mut()
            .map(|c| c.command())
            .collect()
    }
}

// Placeholder function for building behavior trees (Task 5.4)
// In a real implementation, this would create complex trees based on strategy, roles, etc.
fn build_player_bt(
    _player_id: PlayerId,
    _world: &WorldData,
    _settings: &ExecutorSettings,
) -> Box<dyn BehaviorNode> {
    use crate::behavior_tree::{ActionNode, SelectNode, SequenceNode};
    use crate::roles::skills::GoToPosition; // Example skill
    use dies_core::Vector2;

    // Example: A simple BT that makes the player go to (0,0)
    // More complex logic would be needed here.
    // For instance, different BTs for goalie vs. field player.
    let go_to_origin = ActionNode::new(
        Box::new(GoToPosition::new(Vector2::new(0.0, 0.0))),
        Some("GoToOrigin".to_string()),
    );
    Box::new(SelectNode::new(
        vec![Box::new(go_to_origin)],
        Some("RootSelect".to_string()),
    ))
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
