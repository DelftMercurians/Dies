use serde::{Deserialize, Serialize};

use crate::Vector2;

use super::TeamColor;

/// The game state, as reported by the referee.
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
#[serde(tag = "type", content = "data")]
pub enum GameStateType {
    #[default]
    Unknown,
    Halt,
    Timeout,
    Stop,
    PrepareKickoff,
    /// Automatic ball placement in progress with the given target location
    BallPlacement(Vector2),
    PreparePenalty,
    Kickoff,
    FreeKick,
    Penalty,
    PenaltyRun,
    Run,
}

impl std::fmt::Display for GameStateType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            GameStateType::Unknown => "Unknown".to_string(),
            GameStateType::Halt => "Halt".to_string(),
            GameStateType::Timeout => "Timeout".to_string(),
            GameStateType::Stop => "Stop".to_string(),
            GameStateType::PrepareKickoff => "PrepareKickoff".to_string(),
            GameStateType::BallPlacement(_) => "BallPlacement".to_string(),
            GameStateType::PreparePenalty => "PreparePenalty".to_string(),
            GameStateType::Kickoff => "Kickoff".to_string(),
            GameStateType::FreeKick => "FreeKick".to_string(),
            GameStateType::Penalty => "Penalty".to_string(),
            GameStateType::PenaltyRun => "PenaltyRun".to_string(),
            GameStateType::Run => "Run".to_string(),
        };
        write!(f, "{}", str)
    }
}

impl std::hash::Hash for GameStateType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.to_string().hash(state);
    }
}

impl PartialEq for GameStateType {
    fn eq(&self, other: &Self) -> bool {
        self.to_string() == other.to_string()
    }
}

impl Eq for GameStateType {}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GameState {
    /// The state of current game
    pub game_state: GameStateType,
    /// The team (if any) currently operating in asymmetric states
    pub operating_team: Option<TeamColor>,
}
