use dies_protos::ssl_gc_referee_message::referee::Command;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Copy)]
pub enum GameState {
    Halt,
    UNKNOWN,
    Timeout,
    Stop,
    PrepareKickoff,
    BallReplacement,
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    Penalty_Run,
    Run,
}


#[derive(Debug, Clone, Copy)]
pub struct GameStateTracker {
    game_state: GameState,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameState::UNKNOWN,
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
                    self.game_state.clone()
                }
            }
            Command::FORCE_START => GameState::Run,
            Command::PREPARE_KICKOFF_YELLOW => GameState::PrepareKickoff,
            Command::PREPARE_KICKOFF_BLUE => GameState::PrepareKickoff,
            Command::PREPARE_PENALTY_YELLOW => GameState::PreparePenalty,
            Command::PREPARE_PENALTY_BLUE => GameState::PreparePenalty,
            Command::DIRECT_FREE_YELLOW | Command::DIRECT_FREE_BLUE => {
                if(self.game_state == GameState::Stop) {
                    GameState::FreeKick
                } else {
                    self.game_state.clone()
                }
            },
            Command::TIMEOUT_YELLOW => GameState::Timeout,
            Command::TIMEOUT_BLUE => GameState::Timeout,
            Command::BALL_PLACEMENT_YELLOW | Command::BALL_PLACEMENT_BLUE => GameState::BallReplacement,
            _ => {self.game_state.clone()}
        };

        return self.game_state.clone();

    }

    pub fn get_game_state(&self) -> GameState {
        self.game_state.clone()
    }

    pub fn set_game_state(&mut self, game_state: GameState) {
        self.game_state = game_state;
    }
}
