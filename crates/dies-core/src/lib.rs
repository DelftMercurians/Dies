mod angle;
mod controller_settings;
mod executor_info;
mod geom;
mod player;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use controller_settings::*;
pub use executor_info::*;
pub use geom::*;
pub use player::*;
pub use world::*;

use serde::Serialize;
use typeshare::typeshare;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

/// Setup for a player in a scenario.
#[derive(Clone, Debug, Serialize)]
#[typeshare]
pub struct PlayerPlacement {
    /// Initial position of the player. If `None`, any position is acceptable.
    pub position: Option<Vector2>,
    /// Initial yaw of the player. If `None`, any yaw is acceptable.
    pub yaw: Option<Angle>,
}

/// Setup for the ball in a scenario.
#[derive(Clone, Debug, Serialize)]
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
#[derive(Clone, Debug, Serialize)]
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
