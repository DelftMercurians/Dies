use dies_core::{PlayerFeedbackMsg, VisionMsg, WorldFrame};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct GameSetup {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
/// Message pushed from the server to the UI
pub enum UiPushMsg {
    World(WorldFrame),
    // Vision(VisionMsg),
    // Basestation(PlayerFeedbackMsg),
    Debug,
    StrategyConsoleLine(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
/// Command sent from the UI to the server
pub enum UiCmd {
    StartGame(GameSetup),
    StopGame,
    PauseGame,
    ConnectVision,
    ConnectGc,
    ConnectBasestation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub enum GameStatus {
    Stopped,
    Running,
    Paused,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum ConnectionStatus {
    None,
    Connected,
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct UiState {
    game_status: GameStatus,
    vision_status: ConnectionStatus,
    gc_status: ConnectionStatus,
    bs_status: ConnectionStatus,
}

impl Default for UiState {
    fn default() -> Self {
        Self {
            game_status: GameStatus::Stopped,
            vision_status: ConnectionStatus::None,
            gc_status: ConnectionStatus::None,
            bs_status: ConnectionStatus::None,
        }
    }
}
