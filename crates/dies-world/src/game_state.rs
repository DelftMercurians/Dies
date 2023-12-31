use crate::BallData;
use dies_core::GameState;
use dies_protos::ssl_gc_referee_message::referee::Command;
use nalgebra::Vector3;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct GameStateTracker {
    game_state: GameState,
    prev_state: GameState,
    new_state_movement: GameState,
    new_state_timeout: GameState,
    init_ball_pos: Vector3<f32>,
    start: Instant,
    timeout: u64,
    is_outdated: bool,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameState::Unknown,
            prev_state: GameState::Unknown,
            new_state_movement: GameState::Unknown,
            new_state_timeout: GameState::Unknown,
            init_ball_pos: Vector3::new(0.0, 0.0, 0.0),
            start: Instant::now(),
            timeout: 0,
            is_outdated: true,
        }
    }

    pub fn update(&mut self, command: &Command) -> GameState {
        self.game_state = match command {
            Command::HALT => GameState::Halt,
            Command::STOP => GameState::Stop,
            Command::NORMAL_START => {
                if self.game_state == GameState::PrepareKickoff {
                    GameState::Kickoff
                } else if self.game_state == GameState::PreparePenalty {
                    GameState::Penalty
                } else {
                    self.game_state
                }
            }
            Command::FORCE_START => GameState::Run,
            Command::PREPARE_KICKOFF_YELLOW => GameState::PrepareKickoff,
            Command::PREPARE_KICKOFF_BLUE => GameState::PrepareKickoff,
            Command::PREPARE_PENALTY_YELLOW => GameState::PreparePenalty,
            Command::PREPARE_PENALTY_BLUE => GameState::PreparePenalty,
            Command::DIRECT_FREE_YELLOW | Command::DIRECT_FREE_BLUE => {
                if self.game_state == GameState::Stop {
                    GameState::FreeKick
                } else {
                    self.game_state
                }
            }
            Command::TIMEOUT_YELLOW => GameState::Timeout,
            Command::TIMEOUT_BLUE => GameState::Timeout,
            Command::BALL_PLACEMENT_YELLOW | Command::BALL_PLACEMENT_BLUE => {
                GameState::BallReplacement
            }
            _ => self.game_state,
        };

        return self.game_state;
    }

    pub fn start_ball_movement_check(&mut self, ball_pos: Vector3<f32>, timeout: u64) {
        if self.is_outdated == false {
            return;
        }
        self.init_ball_pos = ball_pos.clone();
        self.start = Instant::now();
        self.timeout = timeout;
        self.is_outdated = false;
        self.prev_state = self.game_state;
        self.new_state_movement = match self.game_state {
            GameState::Kickoff | GameState::FreeKick => GameState::Run,
            GameState::Penalty => GameState::PenaltyRun,
            _ => self.game_state,
        };
        self.new_state_timeout = match self.game_state {
            GameState::Kickoff | GameState::FreeKick => GameState::Run,
            GameState::Penalty => GameState::Stop,
            _ => self.game_state,
        };
    }

    pub fn update_ball_movement_check(&mut self, ball_data: Option<&BallData>) -> GameState {
        let p = self.init_ball_pos.clone();
        if self.is_outdated == true || ball_data.is_none() {
            return self.game_state;
        } else if self.prev_state != self.game_state {
            self.is_outdated = true;
        } else if self.start.elapsed().as_secs() >= self.timeout {
            self.game_state = self.new_state_timeout;
            self.is_outdated = true;
        } else {
            let distance = (ball_data.unwrap().position - p).norm();
            let velocity = ball_data.unwrap().velocity.norm();
            if distance > 100.0 && velocity > 100.0 {
                self.game_state = self.new_state_movement;
                self.is_outdated = true;
            }
            return self.game_state;
        }
        return self.game_state;
    }

    pub fn get_game_state(&self) -> GameState {
        self.game_state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_state::GameState::Stop;
    use dies_protos::ssl_gc_referee_message::referee::Command::FORCE_START;

    #[test]
    fn test_new_game_state_tracker() {
        let tracker = GameStateTracker::new();
        assert_eq!(tracker.get_game_state(), GameState::Unknown);
    }

    #[test]
    fn test_normal_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&Command::HALT);
        assert_eq!(tracker.get_game_state(), GameState::Halt);
        tracker.update(&Command::STOP);
        assert_eq!(tracker.get_game_state(), Stop);
        tracker.update(&FORCE_START);
        assert_eq!(tracker.get_game_state(), GameState::Run);
        tracker.update(&Command::PREPARE_KICKOFF_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::PrepareKickoff);
        tracker.update(&Command::PREPARE_PENALTY_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::PreparePenalty);
        tracker.update(&Command::TIMEOUT_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::Timeout);
        tracker.update(&Command::BALL_PLACEMENT_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::BallReplacement);
    }

    #[test]
    fn test_only_once_update() {
        let mut tracker = GameStateTracker::new();
        tracker.update(&Command::PREPARE_KICKOFF_YELLOW);
        assert_eq!(tracker.get_game_state(), GameState::PrepareKickoff);
        tracker.update(&Command::NORMAL_START);
        assert_eq!(tracker.get_game_state(), GameState::Kickoff);
        tracker.update(&Command::NORMAL_START);
        assert_eq!(tracker.get_game_state(), GameState::Kickoff);

        tracker.update(&Command::STOP);
        assert_eq!(tracker.get_game_state(), Stop);
        tracker.update(&Command::DIRECT_FREE_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::FreeKick);
        tracker.update(&Command::DIRECT_FREE_BLUE);
        assert_eq!(tracker.get_game_state(), GameState::FreeKick);
    }
}
