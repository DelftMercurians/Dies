mod angle;
mod geom;
mod player;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use geom::*;
pub use player::*;
use typeshare::typeshare;
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

/// Setup for a player in a scenario.
#[derive(Clone, Debug, Deserialize)]
#[typeshare]
pub struct PlayerPlacement {
    /// Initial position of the player. If `None`, any position is acceptable.
    pub position: Option<Vector2>,
    /// Initial yaw of the player. If `None`, any yaw is acceptable.
    pub yaw: Option<Angle>,
}

/// Setup for the ball in a scenario.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum BallPlacement {
    /// Ball is placed at a specific position.
    Position(Vector3),
    /// Ball is placed at any position.
    AnyPosition,
    /// No ball is required.
    NoBall,
}

/// Information about a scenario.
#[derive(Clone, Debug, Deserialize)]
#[typeshare]
pub struct ScenarioInfo {
    pub own_player_placements: Vec<PlayerPlacement>,
    pub opponent_player_placements: Vec<PlayerPlacement>,
    pub ball_placement: BallPlacement,
    /// Position tolerance for player and ball positions in mm.
    pub tolerance: f64,
    /// Yaw tolerance for players in rad
    pub yaw_tolerance: f64,
}
