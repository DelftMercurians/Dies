use std::collections::HashMap;

use crate::player_controller::PlayerController;
use dies_core::GameState;
use dies_core::{PlayerCmd, WorldData};
use nalgebra::{Vector2, Vector3};
use std::cell::RefCell;
use std::rc::Rc;

pub struct HaltController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
}
impl HaltController {
    /// everyone stops, notice that this only interrupts the players, so if the game
    ///recovers the players will head to their original goal.
    pub fn new(players: Rc<RefCell<HashMap<u32, PlayerController>>>) -> Self {
        Self { players }
    }

    pub fn update(&mut self) -> Vec<PlayerCmd> {
        self.players
            .borrow()
            .iter()
            .map(|(id, _)| PlayerCmd::zero(*id))
            .collect()
    }
}

pub struct StopController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
}
impl StopController {
    /// After generating the commands, everyone's speed is capped at 1.5m/s.
    /// and if the player is less than 500mm from the ball, the goal is set to the point 500mm away from the ball.
    /// and in the opposite of the ball's speed direction.
    /// dribbler is also stopped.
    /// we only issue commands for players in the current frame,
    /// and we assume the world covers every player's state in a long enough time.
    pub fn new(players: Rc<RefCell<HashMap<u32, PlayerController>>>) -> Self {
        Self { players }
    }

    pub fn update(&mut self, world_data: WorldData) -> Vec<PlayerCmd> {
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;
        let ball_speed = world_data.ball.clone().unwrap().velocity;
        let ball_pos_v2 = Vector2::new(ball_pos.x, ball_pos.y);
        let ball_speed_v2 = Vector2::new(ball_speed.x, ball_speed.y);
        let ball_speed_norm = ball_speed_v2.norm();
        let mut players_borrow_mut = self.players.borrow_mut();

        for playerData in world_data.own_players {
            let id = playerData.id;
            let player_controller = players_borrow_mut.get_mut(&id).unwrap();

            if ball_speed_norm > 0.1 {
                let ball_speed_dir: Vector2<f32> = ball_speed_v2 / ball_speed_norm;
                let target_pos: Vector2<f32> = ball_pos_v2 - ball_speed_dir * 500.0;
                player_controller.set_target_pos(target_pos);
            } else {
                let target_pos_dir: Vector2<f32> = (playerData.position - ball_pos_v2).normalize();
                let target_pos: Vector2<f32> = ball_pos_v2 + target_pos_dir * 500.0;
                player_controller.set_target_pos(target_pos);
            }
            let mut cmd = player_controller.update(&playerData, false, false);
            let player_speed = (cmd.sx * cmd.sx + cmd.sy * cmd.sy).sqrt();
            if player_speed > 1500.0 {
                cmd.sx = cmd.sx * 1500.0 / player_speed;
                cmd.sy = cmd.sy * 1500.0 / player_speed;
            }
            commands.push(cmd);
        }
        commands
    }
}

pub struct TeamController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
    halt_controller: HaltController,
    stop_controller: StopController,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new() -> Self {
        let players = Rc::new(RefCell::new(HashMap::new()));
        Self {
            halt_controller: HaltController::new(Rc::clone(&players)),
            stop_controller: StopController::new(Rc::clone(&players)),
            players,
        }
    }

    /// Set the target position for the player with the given ID.
    pub fn set_target_pos(&mut self, id: u32, setpoint: Vector2<f32>) {
        let mut players = self.players.borrow_mut();
        if let Some(player) = players.get_mut(&id) {
            player.set_target_pos(setpoint);
        }
    }

    /// Update the controllers with the current state of the players.
    pub fn update(&mut self, world_data: WorldData) -> Vec<PlayerCmd> {
        let mut players = self.players.borrow_mut();
        let tracked_ids: Vec<u32> = world_data.own_players.iter().map(|p| p.id).collect();
        for player in &world_data.own_players {
            let id = player.id;
            let player_controller = players
                .entry(id)
                .or_insert_with(|| PlayerController::new(id));
            player_controller.update_current_pos(&player);
        }
        for (id, player_controller) in players.iter_mut() {
            if !tracked_ids.contains(id) {
                player_controller.increment_frame_missings();
            }
        }
        match world_data.current_game_state {
            GameState::Halt | GameState::Timeout => self.halt_controller.update(),
            GameState::Stop => self.stop_controller.update(world_data),
            _ => Vec::new(),
        }
    }
}
