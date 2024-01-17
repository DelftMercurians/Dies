//! Defines the interface for an environment.
//!
//! Environments should provide two types: one implementing [`EnvSender`] and one
//! implementing [`EnvReceiver`]. The [`EnvSender`] is used to send commands to the
//! environment, while the [`EnvReceiver`] is used to receive events from the
//! environment.
//!
//! Environments should also provided a function that creates a pair of sender and
//! receiver:
//!
//! ```no_run
//! use dies_core::env::{EnvSender, EnvReceiver};
//!
//! pub fn create_env() -> (impl EnvSender, impl EnvReceiver) {
//!    todo!()
//! }
//! ```

// use std::net::TcpStream;

use crate::{GcRefereeMsg, PlayerCmd, VisionMsg};
use anyhow::Result;

use dies_protos::ssl_gc_rcon_team::TeamToController;

/// An event from the environment.
#[derive(Debug, Clone)]
pub enum EnvEvent {
    /// A packet from ssl-vision
    VisionMsg(VisionMsg),
    /// A packet from ssl-game-controller
    GcRefereeMsg(GcRefereeMsg),
}

/// A trait for the configuration of an environment.
pub trait EnvConfig {
    /// Create a pair of sender and receiver for the environment.
    fn build(self) -> Result<(Box<dyn EnvSender>, Box<dyn EnvReceiver>)>;
}

/// A trait for the sender side of an environment.
pub trait EnvSender: Send {
    /// Send a command to a player
    fn send_player(&self, cmd: PlayerCmd) -> anyhow::Result<()>;

    /// Send a command to the game controller
    fn send_gc(&mut self, team: TeamToController) -> anyhow::Result<()>;
}

/// A trait for the receiver side of an environment.
pub trait EnvReceiver: Send {
    /// Receive an event from the environment.
    ///
    /// This method blocks until an event is received.
    fn recv(&mut self) -> Result<EnvEvent>;
}
