mod angle;
mod debug_info;
mod executor_info;
mod executor_settings;
mod geom;
mod math;
mod player;
mod sides;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use debug_info::*;
pub use executor_info::*;
pub use executor_settings::*;
pub use geom::*;
pub use math::*;
pub use player::*;
use serde::{Deserialize, Serialize};
pub use sides::*;
use typeshare::typeshare;
pub use world::*;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

/// Command to modify the simulator state.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum SimulatorCmd {
    ApplyBallForce {
        force: Vector2,
    },
    TeleportRobot {
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    },
    AddRobot {
        team_color: TeamColor,
        player_id: PlayerId,
        position: Vector2,
        yaw: Angle,
    },
    RemoveRobot {
        team_color: TeamColor,
        player_id: PlayerId,
    },
}
