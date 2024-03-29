use std::collections::HashMap;

use dies_core::{PlayerCmd, PlayerData, WorldData};
use nalgebra::Vector2;
use std::cell::RefCell;
use std::rc::Rc;
use dies_core::GameState;
use crate::player_controller::PlayerController;

pub struct HaltController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
}
impl HaltController {
    pub fn new(players: Rc<RefCell<HashMap<u32, PlayerController>>>) -> Self {
        Self { players }
    }

    pub fn update(&mut self, world_data: WorldData) -> Vec<PlayerCmd> {
        self.players.borrow().iter().map(|(id, _)|PlayerCmd::zero(*id)).collect()
    }
}


pub struct TeamController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
    halt_controller: HaltController,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new() -> Self {
        let players = Rc::new(RefCell::new(HashMap::new()));
        Self {
            halt_controller: HaltController::new(Rc::clone(&players)),
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
        for player in world_data.own_players {
            let id = player.id;
            let player_controller = players.entry(id).or_insert_with(|| PlayerController::new(id));
            player_controller.update_current_pos(&player);
        }
        for (id, player_controller) in players.iter_mut() {
            if !tracked_ids.contains(id) {
                player_controller.increment_frame_missings();
            }
        }
        match world_data.current_game_state {
            GameState::Halt => self.halt_controller.update(world_data),
            _ => Vec::new(),
        }

    }
}
