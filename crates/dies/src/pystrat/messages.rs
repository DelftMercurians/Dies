use serde::{Deserialize, Serialize};

/// A message to a strategy process
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum StratMsg {
    /// Terminates the strategy process
    Term,

    // A debug message
    Hello {
        message: String,
    },
}

/// A message from a strategy process
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum StratCmd {
    /// A debug message
    Debug { message: String },
}
