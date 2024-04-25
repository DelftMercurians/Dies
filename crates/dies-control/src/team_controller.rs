use std::collections::HashMap;

use crate::player_controller::PlayerController;
use dies_core::GameState;
use dies_core::{PlayerCmd, WorldData};
use nalgebra::Vector2;
use std::cell::RefCell;
use std::cmp::PartialEq;
use std::f32::consts::PI;
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

        for player_data in world_data.own_players {
            let id = player_data.id;
            let player_controller = players_borrow_mut.get_mut(&id).unwrap();
            if (player_data.position - ball_pos_v2).norm() < 500.0 {
                // if the ball has a speed, i.e >0.1mm/s
                if ball_speed_norm > 0.1 {
                    let ball_speed_dir: Vector2<f32> = ball_speed_v2 / ball_speed_norm;
                    let target_pos: Vector2<f32> = ball_pos_v2 - ball_speed_dir * 500.0;
                    player_controller.set_target_pos(target_pos);
                } else {
                    let target_pos_dir: Vector2<f32> = (player_data.position - ball_pos_v2).normalize();
                    let target_pos: Vector2<f32> = ball_pos_v2 + target_pos_dir * 500.0;
                    player_controller.set_target_pos(target_pos);
                }
            }
            let mut cmd = player_controller.update(false, false);
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

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum PlayerStatus {
    NoGoal,
    Ongoing,
    Accomplished,
}

pub struct KickOffController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
    assigned_player: Option<(u32, PlayerStatus)>,
    pos_assigned: HashMap<u32, Vector2<f32>>,
    step: f32,
}


impl KickOffController {
    pub fn new (players: Rc<RefCell<HashMap<u32, PlayerController>>>) -> Self {
        Self { players, assigned_player: None,
            pos_assigned: HashMap::new(),
            step: 500.0
        }
    }

    /// our prekick-off -> move to our half of the field assign 1 robot in the circle
    /// opponent's prekick-off ->  move to our half of the field
    /// our kick-off -> the assigned player kick the ball
    pub fn update(&mut self, world_data: WorldData) -> Vec<PlayerCmd> {
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;
        let mut players_borrow_mut = self.players.borrow_mut();


        if world_data.current_game_state.game_state == GameState::PrepareKickoff {

            // if us attacking, assign a player to the ball
            if world_data.current_game_state.us_operating && self.assigned_player.is_none() && !world_data.own_players.is_empty() {
                let aid = world_data.own_players[0].id;
                self.assigned_player = Some((aid, PlayerStatus::NoGoal));
                let controller = players_borrow_mut.get_mut(&aid).unwrap();
                // we go to a fixed position inside the circle
                controller.set_target_pos(Vector2::new(100.0, 0.0));
                self.pos_assigned.insert(aid, Vector2::new(100.0, 0.0));
            }

            // move our robot to our half of the field
            let x_coord = (world_data.field_geom.clone().unwrap().field_length as f32) / 4.0;
            for player in world_data.own_players {
                let id = player.id;
                let controller = players_borrow_mut.get_mut(&id).unwrap();
                if !self.pos_assigned.contains_key(&id) {
                    let y_pos = match self.pos_assigned.len()%2 {
                        0 => ((self.pos_assigned.len()/2) as f32) * self.step,
                        1 => {
                            let offset: i32 = (self.pos_assigned.len() as i32 + 1) / 2;
                            - (offset as f32) * self.step
                        },
                        _ => 0.0
                    };
                    let target_pos = Vector2::new(x_coord, y_pos);
                    controller.set_target_pos(target_pos);
                    self.pos_assigned.insert(id, target_pos);
                }
                let cmd = controller.update(false, false);
                commands.push(cmd);
            }
        }
        else if let Some((aid, status)) = self.assigned_player.as_ref() {
            //find the balldata of the assigned player, in the optional variable
            let playerdata = world_data.own_players.iter().find(|p| p.id == *aid);

            let controller = players_borrow_mut.get_mut(aid).unwrap();
            let mut kick = false;
            let new_status = match status {
                PlayerStatus::NoGoal => {
                    controller.set_target_pos(ball_pos.xy());
                    controller.set_target_heading(-PI);
                    PlayerStatus::Ongoing
                },
                PlayerStatus::Ongoing => {
                    if playerdata.is_some() {
                        let player = playerdata.unwrap();
                        if (player.position - ball_pos.xy()).norm() < 100.0 &&
                            player.orientation + PI < 0.1 {
                            PlayerStatus::Accomplished
                        } else {
                            PlayerStatus::Ongoing
                        }
                    } else {
                        PlayerStatus::Ongoing
                    }
                },
                PlayerStatus::Accomplished => {
                    kick = true;
                    PlayerStatus::Accomplished
                }
            };

            let cmd = controller.update(false, kick);
            commands.push(cmd);
            self.assigned_player = Some((*aid, new_status));
        }
        commands
    }
}


#[derive(Clone, Debug, PartialEq, Copy)]
pub enum BallReplacementStatus {
    NoGoal,
    BeforeManipulation,
    AfterManipulation,
    Accomplished,
}


pub struct BallReplacementController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
    assigned_player: Option<(u32, BallReplacementStatus)>,
    assigned_pos: Option<Vector2<f32>>,
}

impl BallReplacementController {
    pub fn new(players: Rc<RefCell<HashMap<u32, PlayerController>>>) -> Self {
        Self {
            players,
            assigned_player: None,
            assigned_pos: None,
        }
    }
    /// assign the nearest player to the ball to replace the ball
    /// it will move to the ball and heading to it(before manipulation)
    /// then it starts dribbling and move to the designated position(after manipulation)
    /// then it stops the dribbler and the task is accomplished, afterwards the robot step back 0.05m
    pub fn update(&mut self, world_data: WorldData, designated_pos: Vector2<f32>) -> Vec<PlayerCmd>{
        self.assigned_pos = Some(designated_pos);
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;
        let mut players_borrow_mut = self.players.borrow_mut();

        if self.assigned_player.is_none() {
            //assign the nearest player to the ball
            let mut min_distance = f32::MAX;
            let mut nearest_player = None;
            for player in &world_data.own_players {
                let distance = (player.position - ball_pos.xy()).norm();
                if distance < min_distance {
                    min_distance = distance;
                    nearest_player = Some(player.id);
                }
            }
            if let Some(np) = nearest_player {
                self.assigned_player = Some((np, BallReplacementStatus::NoGoal));
            }
        }

        // if the ball is within 85mm of the robot, and the abs relative angle is less than PI/4,
        // assume robot is manipulating the ball
        let (aid, status) = self.assigned_player.as_ref().unwrap();
        
        // if aid is not in the current playerdata
        if world_data.own_players.iter().find(|p| p.id == *aid).is_none() {
            return commands;
        }
        
        let controller = players_borrow_mut.get_mut(aid).unwrap();
        
        
        
        
        commands
    }

}






pub struct TeamController {
    players: Rc<RefCell<HashMap<u32, PlayerController>>>,
    halt_controller: HaltController,
    stop_controller: StopController,
    kick_off_controller: KickOffController,
}

impl Default for TeamController {
    fn default() -> Self {
        Self::new()
    }
}

impl TeamController {
    /// Create a new team controller.
    pub fn new() -> Self {
        let players = Rc::new(RefCell::new(HashMap::new()));
        Self {
            halt_controller: HaltController::new(Rc::clone(&players)),
            stop_controller: StopController::new(Rc::clone(&players)),
            kick_off_controller: KickOffController::new(Rc::clone(&players)),
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
            player_controller.update_current_pos(player);
        }
        for (id, player_controller) in players.iter_mut() {
            if !tracked_ids.contains(id) {
                player_controller.increment_frame_missings();
            }
        }

        match world_data.current_game_state.game_state {
            GameState::Halt | GameState::Timeout => self.halt_controller.update(),
            GameState::Stop => self.stop_controller.update(world_data),
            GameState::PrepareKickoff | GameState::Kickoff => {
                self.kick_off_controller.update(world_data)
            }
            GameState::BallReplacement(pos) => {
                if world_data.current_game_state.us_operating {
                    self.stop_controller.update(world_data)
                } else {
                    self.stop_controller.update(world_data)
                }
            }
            _ => Vec::new(),
        }
    }
}
