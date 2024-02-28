use dies_core::{PlayerCmd, PlayerPosCmd, WorldData};
use serde::{Deserialize, Serialize};

/// A message to the runtime
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum RuntimeMsg {
    /// Updated world state
    World(WorldData),
    /// Terminates the runtime process
    Term,
}

/// A message from the runtime
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum RuntimeEvent {
    /// A position command to one of our players
    PlayerPosCmd(PlayerPosCmd),
    /// A command to one of our players
    PlayerCmd(PlayerCmd),
    /// A debug message
    Debug {
        /// The debug message
        msg: String,
    },
    /// A crash message
    Crash {
        /// The error message
        msg: String,
    },
    /// A ping message
    Ping,
}
