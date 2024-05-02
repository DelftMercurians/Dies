use std::collections::{HashMap, HashSet};

use crate::player_controller::{KickerControlInput, PlayerControlInput, PlayerController};
use dies_core::GameState;
use dies_core::{PlayerCmd, WorldData};
use nalgebra::Vector2;
use std::cmp::PartialEq;
use std::f32::consts::PI;

/// Everyone stops, notice that this only interrupts the players, so if the game
/// recovers the players will head to their original goal.
pub struct HaltController;

impl HaltController {
    pub fn new() -> Self {
        Self
    }

    pub fn update(&mut self, world: &WorldData) -> Vec<PlayerControlInput> {
        world
            .own_players
            .iter()
            .map(|p| PlayerControlInput::new(p.id))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum PlayerStatus {
    NoGoal,
    Ongoing,
    Accomplished,
}

pub struct KickOffController {
    assigned_player: Option<(u32, PlayerStatus)>,
    pos_assigned: HashMap<u32, Vector2<f32>>,
    step: f32,
}

impl KickOffController {
    pub fn new() -> Self {
        Self {
            assigned_player: None,
            pos_assigned: HashMap::new(),
            step: 500.0,
        }
    }

    /// our prekick-off -> move to our half of the field assign 1 robot in the circle
    /// opponent's prekick-off ->  move to our half of the field
    /// our kick-off -> the assigned player kick the ball
    pub fn update(&mut self, world_data: &WorldData) -> Vec<PlayerControlInput> {
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;

        if world_data.current_game_state.game_state == GameState::PrepareKickoff {
            // if us attacking, assign a player to the ball
            if world_data.current_game_state.us_operating
                && self.assigned_player.is_none()
                && !world_data.own_players.is_empty()
            {
                let aid = world_data.own_players[0].id;
                self.assigned_player = Some((aid, PlayerStatus::NoGoal));
                // we go to a fixed position inside the circle
                let pos = Vector2::new(100.0, 0.0);
                commands.push(PlayerControlInput::new(aid).with_position(pos));
                self.pos_assigned.insert(aid, pos);
            }

            // move our robot to our half of the field
            let x_coord = (world_data.field_geom.clone().unwrap().field_length as f32) / 4.0;
            for player in world_data.own_players.iter() {
                let id = player.id;
                if !self.pos_assigned.contains_key(&id) {
                    let y_pos = match self.pos_assigned.len() % 2 {
                        0 => ((self.pos_assigned.len() / 2) as f32) * self.step,
                        1 => {
                            let offset: i32 = (self.pos_assigned.len() as i32 + 1) / 2;
                            -(offset as f32) * self.step
                        }
                        _ => 0.0,
                    };
                    let target_pos = Vector2::new(x_coord, y_pos);
                    commands.push(PlayerControlInput::new(id).with_position(target_pos));
                    self.pos_assigned.insert(id, target_pos);
                }
            }
        } else if let Some((aid, status)) = self.assigned_player.as_ref() {
            //find the balldata of the assigned player, in the optional variable
            let playerdata = world_data.own_players.iter().find(|p| p.id == *aid);

            let new_status = match status {
                PlayerStatus::NoGoal => {
                    commands.push(
                        PlayerControlInput::new(*aid)
                            .with_position(ball_pos.xy())
                            .with_orientation(-PI),
                    );
                    PlayerStatus::Ongoing
                }
                PlayerStatus::Ongoing => {
                    if playerdata.is_some() {
                        let player = playerdata.unwrap();
                        if (player.position - ball_pos.xy()).norm() < 10.0
                            && player.orientation + PI < 0.1
                        {
                            PlayerStatus::Accomplished
                        } else {
                            PlayerStatus::Ongoing
                        }
                    } else {
                        PlayerStatus::Ongoing
                    }
                }
                PlayerStatus::Accomplished => {
                    commands.push(
                        PlayerControlInput::new(*aid)
                            .with_dribbling(1.0)
                            .with_kicker(KickerControlInput::Kick),
                    );
                    PlayerStatus::Accomplished
                }
            };

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
    assigned_player: Option<(u32, BallReplacementStatus)>,
    assigned_pos: Option<Vector2<f32>>,
    play_dir_x: f32,
}

impl BallReplacementController {
    pub fn new(play_dir_x: f32) -> Self {
        Self {
            assigned_player: None,
            assigned_pos: None,
            play_dir_x,
        }
    }
    /// assign the nearest player to the ball to replace the ball
    /// it will move to the ball and heading to it(before manipulation)
    /// then it starts dribbling and move to the designated position(after manipulation)
    /// then it stops the dribbler and the task is accomplished, afterwards the robot step back 0.05m
    pub fn update(
        &mut self,
        world_data: &WorldData,
        designated_pos: Vector2<f32>,
    ) -> Vec<PlayerControlInput> {
        self.assigned_pos = Some(designated_pos);
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;

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

        let (aid, status) = self.assigned_player.as_ref().unwrap();

        if let Some(player) = world_data.own_players.iter().find(|p| p.id == *aid) {
            let player_pos = player.position;
            // if the ball is within 80mm of the robot ans within PI/4 angle
            // assume robot is manipulating the ball
            let ball_angle =
                (ball_pos.xy() - player_pos).angle(&Vector2::new(self.play_dir_x, 0.0));
            let is_manipulating = (player_pos - ball_pos.xy()).norm() < 80.0
                && (ball_angle - player.orientation).abs() < PI / 4.0;

            let target_angle =
                (designated_pos - player_pos).angle(&Vector2::new(self.play_dir_x, 0.0));
            let new_status = match status {
                BallReplacementStatus::NoGoal => {
                    commands.push(
                        PlayerControlInput::new(*aid)
                            .with_position(ball_pos.xy())
                            .with_orientation(ball_angle),
                    );
                    BallReplacementStatus::BeforeManipulation
                }
                BallReplacementStatus::BeforeManipulation => {
                    if is_manipulating {
                        commands.push(
                            PlayerControlInput::new(*aid)
                                .with_position(designated_pos)
                                .with_orientation(target_angle)
                                .with_dribbling(1.0),
                        );
                        BallReplacementStatus::AfterManipulation
                    } else {
                        BallReplacementStatus::BeforeManipulation
                    }
                }
                BallReplacementStatus::AfterManipulation => {
                    if (player_pos - designated_pos).norm() < 10.0 {
                        //step back 50mm
                        let opposite_dir = if (player.orientation + PI) > PI {
                            player.orientation + PI - 2.0 * PI
                        } else {
                            player.orientation + PI
                        };

                        commands.push(
                            PlayerControlInput::new(*aid)
                                .with_position(
                                    designated_pos
                                        - (player_pos - designated_pos).normalize() * 50.0,
                                )
                                .with_orientation(opposite_dir),
                        );

                        BallReplacementStatus::Accomplished
                    } else if !is_manipulating {
                        commands.push(
                            PlayerControlInput::new(*aid)
                                .with_position(designated_pos)
                                .with_orientation(target_angle),
                        );
                        BallReplacementStatus::BeforeManipulation
                    } else {
                        commands.push(
                            PlayerControlInput::new(*aid)
                                .with_position(designated_pos)
                                .with_orientation(target_angle),
                        );
                        BallReplacementStatus::AfterManipulation
                    }
                }
                BallReplacementStatus::Accomplished => BallReplacementStatus::Accomplished,
            };

            self.assigned_player = Some((*aid, new_status));
        }
        commands
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum PenaltyKickStatus {
    NoGoal,
    Ongoing,
    AdjustingDirections,
    Accomplished,
}

pub struct PenaltyKickController {
    assigned_player: Option<(u32, PenaltyKickStatus)>,
    pos_assigned: HashMap<u32, Vector2<f32>>,
    kick_angle: Option<f32>,
}

impl PenaltyKickController {
    pub fn new() -> Self {
        Self {
            assigned_player: None,
            pos_assigned: HashMap::new(),
            kick_angle: None,
        }
    }

    /// generate a location to go based on an existing assigned locations
    /// TODO: collision avoidance
    pub fn generate_position(&self, x_range: Vector2<f32>) -> Vector2<f32> {
        // get all the assigned xs and all the assigned ys
        let mut xs: Vec<f32> = [x_range.x, x_range.y].to_vec();
        let mut ys: Vec<f32> = [-3000.0, 3000.0].to_vec();
        for (_, pos) in &self.pos_assigned {
            xs.push(pos.x);
            ys.push(pos.y);
        }

        // sort xs and ys
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // new x/y is the middle of the biggest interval
        let mut new_x = x_range.x;
        let mut new_y = -3000.0;
        let mut max_interval = 0.0;
        for i in 0..xs.len() - 1 {
            let interval = xs[i + 1] - xs[i];
            if interval > max_interval {
                max_interval = interval;
                new_x = xs[i] + interval / 2.0;
            }
        }
        max_interval = 0.0;
        for i in 0..ys.len() - 1 {
            let interval = ys[i + 1] - ys[i];
            if interval > max_interval {
                max_interval = interval;
                new_y = ys[i] + interval / 2.0;
            }
        }
        Vector2::new(new_x, new_y)
    }

    /// our prepenalty kick -> assign 1 robot as the attacker and the others move 1m behind the ball
    /// opponent's prepenalty kick-> goalkeeper on the touch line, others 1m behind the ball
    /// our penalty -> the assigned player kick the ball toward the middle of the goal.
    /// opponent's penalty -> the goalkeeper move onto the intersection point of the ball trajectory
    /// and the touch line.
    /// TODO: add the goalkeeper behaviour if we are defenders.
    pub fn update(&mut self, world_data: &WorldData) -> Vec<PlayerControlInput> {
        let mut commands = Vec::new();
        let ball_pos = world_data.ball.clone().unwrap().position;
        let x_range = if world_data.current_game_state.us_operating {
            Vector2::new(ball_pos.x + 1000.0, 4500.0)
        } else {
            Vector2::new(-4500.0, ball_pos.x - 1000.0)
        };

        if world_data.current_game_state.game_state == GameState::PreparePenalty {
            // if us attacking, assign a player to the ball
            if world_data.current_game_state.us_operating
                && self.assigned_player.is_none()
                && !world_data.own_players.is_empty()
            {
                let aid = world_data.own_players[0].id;
                let player_pos = world_data.own_players[0].position;
                self.assigned_player = Some((aid, PenaltyKickStatus::NoGoal));

                // heading to the nearest 100m place toward the penalty kick mark,with correct heading
                let dir = (ball_pos.xy() - player_pos).angle(&Vector2::new(1.0, 0.0));
                let target_pos = ball_pos.xy() - 100.0 * (ball_pos.xy() - player_pos).normalize();
                commands.push(
                    PlayerControlInput::new(aid)
                        .with_position(target_pos)
                        .with_orientation(dir),
                );
                self.pos_assigned.insert(aid, target_pos);
            }

            for player in &world_data.own_players {
                let id = player.id;
                if !self.pos_assigned.contains_key(&id) {
                    let target_pos = self.generate_position(x_range);
                    commands.push(PlayerControlInput::new(id).with_position(target_pos));
                    self.pos_assigned.insert(id, target_pos);
                }
            }
        } else if world_data.current_game_state.game_state == GameState::Penalty {
            if let Some((aid, status)) = self.assigned_player.as_ref() {
                //find the balldata of the assigned player, in the optional variable
                let playerdata = world_data.own_players.iter().find(|p| p.id == *aid);
                if let Some(player) = playerdata {
                    let player_pos = player.position;
                    let new_status: PenaltyKickStatus = match status {
                        PenaltyKickStatus::NoGoal => {
                            commands.push(
                                PlayerControlInput::new(*aid)
                                    .with_position(ball_pos.xy())
                                    .with_orientation(
                                        (ball_pos.xy() - player_pos).angle(&Vector2::new(1.0, 0.0)),
                                    ),
                            );
                            PenaltyKickStatus::Ongoing
                        }
                        PenaltyKickStatus::Ongoing => {
                            if (player_pos - ball_pos.xy()).norm() < 10.0
                                && (player_pos - ball_pos.xy())
                                    .angle(&Vector2::new(1.0, 0.0))
                                    .abs()
                                    < 0.1
                            {
                                // calculate the direction of the kick
                                let goal_pos = Vector2::new(-3500.0, 0.0);
                                let dir = (goal_pos - ball_pos.xy()).angle(&Vector2::new(1.0, 0.0));
                                self.kick_angle = Some(dir);
                                commands.push(
                                    PlayerControlInput::new(*aid)
                                        .with_position(ball_pos.xy())
                                        .with_orientation(dir),
                                );
                                PenaltyKickStatus::AdjustingDirections
                            } else {
                                PenaltyKickStatus::Ongoing
                            }
                        }
                        PenaltyKickStatus::AdjustingDirections => {
                            commands.push(PlayerControlInput::new(*aid).with_dribbling(1.0));
                            if let Some(_) = self.kick_angle {
                                if (player_pos - ball_pos.xy()).norm() < 10.0
                                    && (player_pos - ball_pos.xy())
                                        .angle(&Vector2::new(1.0, 0.0))
                                        .abs()
                                        < 0.05
                                {
                                    PenaltyKickStatus::Accomplished
                                } else {
                                    PenaltyKickStatus::AdjustingDirections
                                }
                            } else {
                                PenaltyKickStatus::AdjustingDirections
                            }
                        }
                        PenaltyKickStatus::Accomplished => {
                            commands.push(
                                PlayerControlInput::new(*aid)
                                    .with_dribbling(1.0)
                                    .with_kicker(KickerControlInput::Kick),
                            );
                            PenaltyKickStatus::Accomplished
                        }
                    };
                    self.assigned_player = Some((*aid, new_status));
                }
            }
        }
        commands
    }
}

pub struct TeamController {
    player_controllers: HashMap<u32, PlayerController>,
    kick_off_controller: KickOffController,
    ball_replacement_controller: BallReplacementController,
    penalty_kick_controller: PenaltyKickController,
    halt_controller: HaltController,
}

impl TeamController {
    /// Create a new team controller.
    pub fn new(play_dir_x: f32) -> Self {
        Self {
            halt_controller: HaltController::new(),
            kick_off_controller: KickOffController::new(),
            ball_replacement_controller: BallReplacementController::new(play_dir_x),
            penalty_kick_controller: PenaltyKickController::new(),
            player_controllers: HashMap::new(),
        }
    }

    /// Update the controllers with the current state of the players.
    pub fn update(&mut self, world_data: WorldData) {
        // Ensure there is a player controller for every ID
        let detected_ids: HashSet<_> = world_data.own_players.iter().map(|p| p.id).collect();
        for id in detected_ids.iter() {
            if !self.player_controllers.contains_key(id) {
                self.player_controllers
                    .insert(*id, PlayerController::new(*id));
            }
        }

        let mut inputs = match world_data.current_game_state.game_state {
            GameState::Halt | GameState::Timeout => self.halt_controller.update(&world_data),
            GameState::PrepareKickoff | GameState::Kickoff => {
                self.kick_off_controller.update(&world_data)
            }
            GameState::BallReplacement(_) => self
                .ball_replacement_controller
                .update(&world_data, Vector2::new(0.0, 0.0)),
            GameState::PreparePenalty | GameState::Penalty => {
                self.penalty_kick_controller.update(&world_data)
            }
            _ => Vec::new(),
        };

        // If in a stop state, override the inputs
        if world_data.current_game_state.game_state == GameState::Stop {
            inputs = stop_override(world_data.clone(), inputs);
        }

        // Update the player controllers
        for controller in self.player_controllers.values_mut() {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == controller.id());

            if let Some(player_data) = player_data {
                let input = inputs
                    .iter()
                    .find(|i| i.id == controller.id())
                    .cloned()
                    .unwrap_or_else(|| PlayerControlInput::new(controller.id()));

                controller.update(player_data, input);
            } else {
                controller.increment_frames_missings();
            }
        }
    }

    /// Get the currently active commands for the players.
    pub fn commands(&self) -> Vec<PlayerCmd> {
        self.player_controllers
            .values()
            .map(|c| c.command())
            .collect()
    }
}

/// Override the inputs to comply with the stop state.
fn stop_override(
    world_data: WorldData,
    inputs: Vec<PlayerControlInput>,
) -> Vec<PlayerControlInput> {
    let ball_pos = world_data.ball.as_ref().map(|b| b.position.xy());
    let ball_vel = world_data.ball.as_ref().map(|b| b.velocity.xy());
    inputs
        .iter()
        .map(|input| {
            let player_data = world_data
                .own_players
                .iter()
                .find(|p| p.id == input.id)
                .expect("Player not found in world data");

            let mut new_input = input.clone();

            // Cap speed at 1.5m/s
            new_input.velocity = input.velocity.cap_magnitude(1.5);

            // If the player is less than 500mm from the ball, set the goal to the point 500mm away
            // from the ball, in the opposite direction of the ball's speed.
            if let (Some(ball_pos), Some(ball_vel)) = (ball_pos, ball_vel) {
                let dist = (player_data.position - ball_pos).norm();
                if dist < 500.0 {
                    let goal = ball_pos - ball_vel.normalize() * 500.0;
                    new_input.position = Some(goal);
                }
            }

            // Stop dribbler
            new_input.dribbling_speed = 0.0;

            // Disable kick
            new_input.kicker = KickerControlInput::Disarm;

            new_input
        })
        .collect()
}
