use std::time::Instant;

use dies_core::{Vector2, Vector3};
use dies_protos::ssl_gc_referee_message::{referee::Command, Referee};

use crate::{BallFrame, GameState, GameStateType, Team};

#[derive(Debug, Clone, Copy)]
pub struct GameStateTracker {
    /// **NOTE**: The position in `BallPlacement` is in vision coordinates -- the x axis may point in either direction.
    game_state: GameStateType,
    /// **NOTE**: The position in `BallPlacement` is in vision coordinates -- the x axis may point in either direction.
    prev_state: GameStateType,
    /// **NOTE**: The position in `BallPlacement` is in vision coordinates -- the x axis may point in either direction.
    new_state_movement: GameStateType,
    /// **NOTE**: The position in `BallPlacement` is in vision coordinates -- the x axis may point in either direction.
    new_state_timeout: GameStateType,
    init_ball_pos: Vector3,
    start: Instant,
    timeout: u64,
    is_outdated: bool,
    operating_team: Option<Team>,
    last_cmd: Option<Command>,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameStateType::Halt,
            prev_state: GameStateType::Unknown,
            new_state_movement: GameStateType::Unknown,
            new_state_timeout: GameStateType::Unknown,
            init_ball_pos: Vector3::new(0.0, 0.0, 0.0),
            start: Instant::now(),
            timeout: 0,
            is_outdated: true,
            operating_team: None,
            last_cmd: None,
        }
    }

    pub fn update(&mut self, data: &Referee) -> GameStateType {
        let command = data.command();

        if self.last_cmd == Some(command) {
            return self.game_state;
        }
        println!("Referee command: {:?}", command);
        self.last_cmd = Some(command);

        let last_game_state = self.game_state;

        self.game_state = match command {
            Command::HALT => GameStateType::Halt,
            Command::STOP => GameStateType::Stop,
            Command::NORMAL_START => {
                if self.game_state == GameStateType::PrepareKickoff {
                    GameStateType::Kickoff
                } else if self.game_state == GameStateType::PreparePenalty {
                    GameStateType::Penalty
                } else {
                    GameStateType::Run
                }
            }
            Command::FORCE_START => GameStateType::Run,
            Command::PREPARE_KICKOFF_YELLOW => GameStateType::PrepareKickoff,
            Command::PREPARE_KICKOFF_BLUE => GameStateType::PrepareKickoff,
            Command::PREPARE_PENALTY_YELLOW => GameStateType::PreparePenalty,
            Command::PREPARE_PENALTY_BLUE => GameStateType::PreparePenalty,
            Command::DIRECT_FREE_YELLOW
            | Command::DIRECT_FREE_BLUE
            | Command::INDIRECT_FREE_BLUE
            | Command::INDIRECT_FREE_YELLOW => GameStateType::FreeKick,
            Command::TIMEOUT_YELLOW => GameStateType::Timeout,
            Command::TIMEOUT_BLUE => GameStateType::Timeout,
            Command::BALL_PLACEMENT_YELLOW | Command::BALL_PLACEMENT_BLUE => {
                if let Some(pos) = data.designated_position.as_ref() {
                    GameStateType::BallPlacement(Vector2::new(pos.x() as f64, pos.y() as f64))
                } else {
                    log::error!("No position for ball placement");
                    self.game_state
                }
            }
            Command::GOAL_YELLOW | Command::GOAL_BLUE => self.game_state,
        };

        if last_game_state != self.game_state {
            println!("Game state {:?} -> {:?}", last_game_state, self.game_state);
        }

        dies_core::debug_string("last_cmd", format!("{:?}", command));

        self.operating_team = match command {
            Command::PREPARE_KICKOFF_BLUE
            | Command::PREPARE_PENALTY_BLUE
            | Command::DIRECT_FREE_BLUE
            | Command::BALL_PLACEMENT_BLUE => Some(Team::Blue),
            Command::PREPARE_KICKOFF_YELLOW
            | Command::PREPARE_PENALTY_YELLOW
            | Command::DIRECT_FREE_YELLOW
            | Command::BALL_PLACEMENT_YELLOW => Some(Team::Yellow),
            _ => self.operating_team,
        };

        // Reset
        match self.game_state {
            GameStateType::Halt
            | GameStateType::Stop
            | GameStateType::Timeout
            | GameStateType::Run => self.operating_team = None,
            _ => (),
        }

        self.game_state
    }

    pub fn start_ball_movement_check(&mut self, ball_pos: Vector3, timeout: u64) {
        if !self.is_outdated {
            return;
        }
        self.init_ball_pos = ball_pos;
        self.start = Instant::now();
        self.timeout = timeout;
        self.is_outdated = false;
        self.prev_state = self.game_state;
        self.new_state_movement = match self.game_state {
            GameStateType::Kickoff | GameStateType::FreeKick => GameStateType::Run,
            GameStateType::Penalty => GameStateType::PenaltyRun,
            _ => self.game_state,
        };
        self.new_state_timeout = match self.game_state {
            GameStateType::Kickoff | GameStateType::FreeKick => GameStateType::Run,
            GameStateType::Penalty => GameStateType::Stop,
            _ => self.game_state,
        };
    }

    pub fn update_ball_movement_check(&mut self, ball_data: Option<&BallFrame>) -> GameStateType {
        let p = self.init_ball_pos;
        if self.is_outdated || ball_data.is_none() {
            return self.game_state;
        } else if self.prev_state != self.game_state {
            self.is_outdated = true;
        } else if self.start.elapsed().as_secs() >= self.timeout {
            self.game_state = self.new_state_timeout;
            self.is_outdated = true;
        } else {
            let distance = (ball_data.unwrap().position - p).norm();
            let velocity = ball_data.unwrap().velocity.norm();
            if distance > 500.0 && velocity > 5000.0 {
                self.game_state = self.new_state_movement;
                self.is_outdated = true;
            }
            return self.game_state;
        }
        self.game_state
    }

    pub fn get(&self) -> GameState {
        dies_core::debug_string("game_state", format!("{}", self.game_state));

        GameState {
            game_state: self.game_state,
            operating_team: self.operating_team,
        }
    }
}

#[cfg(test)]
mod tests {
    use dies_protos::{
        ssl_gc_referee_message::referee::{Command::FORCE_START, Point},
        MessageField,
    };

    use super::*;
    use crate::game_state::GameStateType::Stop;

    fn referee_msg(command: Command) -> Referee {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg
    }

    fn referee_msg_with_pos(command: Command, pos: Vector2) -> Referee {
        let mut msg = Referee::new();
        msg.set_command(command);
        msg.designated_position = MessageField::some(Point::new());
        msg.designated_position
            .as_mut()
            .unwrap()
            .set_x(pos.x as f32);
        msg.designated_position
            .as_mut()
            .unwrap()
            .set_y(pos.y as f32);
        msg
    }

    #[test]
    fn test_new_game_state_tracker() {
        let tracker = GameStateTracker::new();
        assert_eq!(tracker.get().game_state, GameStateType::Halt);
    }

    #[test]
    fn test_normal_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::HALT));
        assert_eq!(tracker.get().game_state, GameStateType::Halt);
        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get().game_state, Stop);
        tracker.update(&referee_msg(FORCE_START));
        assert_eq!(tracker.get().game_state, GameStateType::Run);
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_BLUE));
        assert_eq!(tracker.get().game_state, GameStateType::PrepareKickoff);
        tracker.update(&referee_msg(Command::PREPARE_PENALTY_BLUE));
        assert_eq!(tracker.get().game_state, GameStateType::PreparePenalty);
        tracker.update(&referee_msg(Command::TIMEOUT_BLUE));
        assert_eq!(tracker.get().game_state, GameStateType::Timeout);
        tracker.update(&referee_msg_with_pos(
            Command::BALL_PLACEMENT_BLUE,
            Vector2::new(0.0, 0.0),
        ));
        assert_eq!(
            tracker.get().game_state,
            GameStateType::BallPlacement(Vector2::new(0.0, 0.0))
        );
    }

    #[test]
    fn test_only_once_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&referee_msg(Command::PREPARE_KICKOFF_YELLOW));
        assert_eq!(tracker.get().game_state, GameStateType::PrepareKickoff);
        tracker.update(&referee_msg(Command::NORMAL_START));
        assert_eq!(tracker.get().game_state, GameStateType::Kickoff);
        tracker.update(&referee_msg(Command::NORMAL_START));
        assert_eq!(tracker.get().game_state, GameStateType::Kickoff);

        tracker.update(&referee_msg(Command::STOP));
        assert_eq!(tracker.get().game_state, Stop);
        tracker.update(&referee_msg(Command::DIRECT_FREE_BLUE));
        assert_eq!(tracker.get().game_state, GameStateType::FreeKick);
        tracker.update(&referee_msg(Command::DIRECT_FREE_BLUE));
        assert_eq!(tracker.get().game_state, GameStateType::FreeKick);
    }
}
