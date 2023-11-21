//! Interfaces for the runtime
//!
//! The runtime is the process that runs the strategy (AI). It could be a Python process,
//! a Rust program, or something else entirely. The runtime is responsible for
//! communicating with the strategy relaying events from the environment.
//!
//! Runtimes need to provide two types: one implementing [`RuntimeSender`] and one
//! implementing [`RuntimeReceiver`]. The [`RuntimeSender`] is used to send messages to the
//! runtime, while the [`RuntimeReceiver`] is used to receive events from the runtime.
//!
//! Runtimes should also provided a function that creates a pair of sender and
//! receiver:
//!
//! ```no_run
//! use dies_core::runtime::{RuntimeSender, RuntimeReceiver};
//!
//! pub fn create_runtime() -> (impl RuntimeSender, impl RuntimeReceiver) {
//!   todo!()
//! }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{PlayerCmd, WorldData};

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
}

/// A trait for the configuration of a runtime.
pub trait RuntimeConfig {
    /// Create a pair of sender and receiver for the runtime.
    fn build(self) -> Result<(Box<dyn RuntimeSender>, Box<dyn RuntimeReceiver>)>;
}

/// The sender side of the runtime
pub trait RuntimeSender: Send {
    /// Send an update to the runtime
    fn send(&mut self, msg: &RuntimeMsg) -> Result<()>;
}

/// The receiver side of the runtime
pub trait RuntimeReceiver: Send {
    /// Receive a message from the runtime.
    ///
    /// This method blocks until a message is received.
    fn recv(&mut self) -> Result<RuntimeEvent>;
}
