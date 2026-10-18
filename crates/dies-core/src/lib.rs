mod angle;
mod avoidance_config;
mod debug_info;
mod executor_info;
mod executor_settings;
mod geom;
mod math;
mod motion_bounds;
mod notebook;
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
pub use notebook::*;
pub use params::*;
pub use player::*;
pub use possession::*;
use serde::{Deserialize, Serialize};
pub use sides::*;
use ts_rs::TS;
pub use tunable::*;
pub use world::*;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

/// TypeScript binding of `Vector2`: nalgebra serializes it as `[x, y]`. Use it
/// as `#[ts(as = "Vector2Ts")]` on `Vector2` fields (nalgebra types can't derive `TS`).
#[derive(TS)]
#[ts(rename = "Vector2", as = "[f64; 2]")]
pub struct Vector2Ts;

/// TypeScript binding of `Vector3`: nalgebra serializes it as `[x, y, z]`. Use it
/// as `#[ts(as = "Vector3Ts")]` on `Vector3` fields.
#[derive(TS)]
#[ts(rename = "Vector3", as = "[f64; 3]")]
pub struct Vector3Ts;

/// Command to modify the simulator state.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type", content = "data")]
pub enum SimulatorCmd {
    ApplyBallForce {
        #[ts(as = "Vector2Ts")]
        force: Vector2,
    },
    TeleportBall {
        #[ts(as = "Vector2Ts")]
        position: Vector2,
    },
    TeleportRobot {
        team_color: TeamColor,
        player_id: PlayerId,
        #[ts(as = "Vector2Ts")]
        position: Vector2,
        yaw: Angle,
    },
    AddRobot {
        team_color: TeamColor,
        player_id: PlayerId,
        #[ts(as = "Vector2Ts")]
        position: Vector2,
        yaw: Angle,
    },
    RemoveRobot {
        team_color: TeamColor,
        player_id: PlayerId,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[serde(tag = "type", content = "data")]
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
        #[ts(as = "Vector2Ts")]
        position: Vector2,
    },
    /// Show a yellow card to a team (lowers its `max_allowed_bots` for 120 s).
    YellowCard {
        team_color: TeamColor,
    },
}

/// A single robot's saved pose in a field snapshot (position + yaw, no velocity).
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
pub struct RobotSnapshot {
    pub id: PlayerId,
    #[ts(as = "Vector2Ts")]
    pub position: Vector2,
    pub yaw: Angle,
}

/// A saved simulator field state: robot poses + ball position, plus an optional
/// game state to seed. Replayed by teleporting everything back into place (see
/// `Simulation::apply_snapshot`) and, when present, forcing the auto-referee into
/// `game_state` for `operating_team`. Both game-state fields are optional so
/// older positional-only snapshots still load.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(optional_fields)]
pub struct FieldSnapshot {
    pub blue: Vec<RobotSnapshot>,
    pub yellow: Vec<RobotSnapshot>,
    #[ts(as = "Option<Vector2Ts>")]
    pub ball: Option<Vector2>,
    /// Game state captured with the snapshot, if any. Seeded on load.
    #[serde(default)]
    pub game_state: Option<GameState>,
    /// Team that operates `game_state` (kickoff/free-kick/penalty/placement).
    /// Ignored for symmetric states (run/stop/halt).
    #[serde(default)]
    pub operating_team: Option<TeamColor>,
}
