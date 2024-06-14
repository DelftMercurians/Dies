mod angle;
mod geom;
mod player;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use geom::*;
pub use player::*;
pub use world::*;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

use serde::Deserialize;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

/// A message from one of our robots to the AI
pub struct PlayerFeedbackMsg {
    /// The robot's ID
    pub id: player::PlayerId,
    /// Capacitor voltage
    pub cap_v: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type")]
pub enum SymScenario {
    Empty,
    SinglePlayerWithoutBall,
    SinglePlayer,
    TwoPlayers,
}
