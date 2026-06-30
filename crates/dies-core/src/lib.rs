mod angle;
mod avoidance_config;
mod debug_info;
mod executor_info;
mod executor_settings;
mod geom;
mod math;
mod motion_bounds;
mod params;
mod player;
mod possession;
mod sides;
mod tunable;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use avoidance_config::*;
pub use debug_info::*;
pub use executor_info::*;
pub use executor_settings::*;
pub use geom::*;
pub use math::*;
pub use motion_bounds::*;
pub use params::*;
pub use player::*;
pub use possession::*;
use serde::{Deserialize, Serialize};
pub use sides::*;
pub use tunable::*;
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
    TeleportBall {
        position: Vector2,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare::typeshare]
pub enum GcSimCommand {
    Stop,
    Halt,
    NormalStart,
    ForceStart,
    KickOff {
        team_color: TeamColor,
    },
    Penalty {
        team_color: TeamColor,
    },
    DirectFree {
        team_color: TeamColor,
    },
    BallPlacement {
        team_color: TeamColor,
        position: Vector2,
    },
}

/// A single robot's saved pose in a field snapshot (position + yaw, no velocity).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct RobotSnapshot {
    pub id: PlayerId,
    pub position: Vector2,
    pub yaw: Angle,
}

/// A saved simulator field state: robot poses + ball position, plus an optional
/// game state to seed. Replayed by teleporting everything back into place (see
/// `Simulation::apply_snapshot`) and, when present, forcing the auto-referee into
/// `game_state` for `operating_team`. Both game-state fields are optional so
/// older positional-only snapshots still load.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct FieldSnapshot {
    pub blue: Vec<RobotSnapshot>,
    pub yellow: Vec<RobotSnapshot>,
    pub ball: Option<Vector2>,
    /// Game state captured with the snapshot, if any. Seeded on load.
    #[serde(default)]
    pub game_state: Option<GameState>,
    /// Team that operates `game_state` (kickoff/free-kick/penalty/placement).
    /// Ignored for symmetric states (run/stop/halt).
    #[serde(default)]
    pub operating_team: Option<TeamColor>,
}
