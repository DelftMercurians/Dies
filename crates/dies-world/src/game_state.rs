use dies_protos::ssl_gc_referee_message::referee::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::{IS_DIV_A, WorldData, WorldTracker};

#[derive(Serialize, Deserialize,Clone, Debug, PartialEq)]
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


#[derive(Debug)]
pub struct GameStateTracker {
    game_state: GameState,
}

impl GameStateTracker {
    pub fn new() -> GameStateTracker {
        GameStateTracker {
            game_state: GameState::UNKNOWN,
        }
    }

    pub fn update(&mut self, command: &Command) {
        self.game_state = match command {
            Command::HALT => GameState::Halt,
            Command::STOP => GameState::Stop,
            Command::NORMAL_START => {
                if self.game_state == GameState::PrepareKickoff {
                    GameState::Kickoff
                } else if self.game_state == GameState::PreparePenalty {
                    GameState::Penalty
                } else {
                    // theoretically this should never happen
                    GameState::UNKNOWN
                }
            }
            Command::FORCE_START => GameState::Run,
            Command::PREPARE_KICKOFF_YELLOW => GameState::PrepareKickoff,
            Command::PREPARE_KICKOFF_BLUE => GameState::PrepareKickoff,
            Command::PREPARE_PENALTY_YELLOW => GameState::PreparePenalty,
            Command::PREPARE_PENALTY_BLUE => GameState::PreparePenalty,
            Command::DIRECT_FREE_YELLOW => GameState::FreeKick,
            Command::DIRECT_FREE_BLUE => GameState::FreeKick,
            Command::TIMEOUT_YELLOW => GameState::Timeout,
            Command::TIMEOUT_BLUE => GameState::Timeout,
            Command::BALL_PLACEMENT_YELLOW | Command::BALL_PLACEMENT_BLUE => GameState::BallReplacement,
            _ => {self.game_state.clone()}
        };

        self.game_state.clone();

    }

    pub fn get_game_state(&self) -> GameState {
        self.game_state.clone()
    }
}
