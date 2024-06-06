mod angle;
mod executor_info;
mod geom;
mod player;
mod world;

pub mod workspace_utils;

pub use angle::*;
pub use executor_info::*;
pub use geom::*;
pub use player::*;
pub use world::*;

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

pub type Scalar = f64;
pub type Vector2 = nalgebra::Vector2<Scalar>;
pub type Vector3 = nalgebra::Vector3<Scalar>;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct PlayerId(u32);

impl PlayerId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

impl std::fmt::Display for PlayerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A command to the kicker of a robot.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum KickerCmd {
    /// Do nothing // TODO: What is this?
    None,
    /// Arm the kicker
    Arm,
    /// Disarm the kicker
    Disarm,
    /// Discharge the kicker without kicking
    Discharge,
    /// Kick the ball
    Kick,
    /// Chip the ball
    Chip,
    /// Power board off
    PowerBoardOff,
}

impl Into<glue::Radio_KickerCommand> for KickerCmd {
    fn into(self) -> glue::Radio_KickerCommand {
        match self {
            KickerCmd::None => glue::Radio_KickerCommand::NONE,
            KickerCmd::Arm => glue::Radio_KickerCommand::ARM,
            KickerCmd::Disarm => glue::Radio_KickerCommand::DISARM,
            KickerCmd::Discharge => glue::Radio_KickerCommand::DISCHARGE,
            KickerCmd::Kick => glue::Radio_KickerCommand::KICK,
            KickerCmd::Chip => glue::Radio_KickerCommand::CHIP,
            KickerCmd::PowerBoardOff => glue::Radio_KickerCommand::POWER_BOARD_OFF,
        }
    }
}

/// A command to one of our players as it will be sent to the robot.
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayerCmd {
    /// The robot's ID
    pub id: PlayerId,
    /// The player's x (forward-backward, with `+` forward) velocity \[m/s]
    pub sx: f64,
    /// The player's y (left-right, with `+` right) velocity \[m/s]
    pub sy: f64,
    /// The player's angular velocity (with `+` counter-clockwise, `-` clockwise) \[rad/s]
    pub w: f64,
    /// The player's dribble speed
    pub dribble_speed: f64,
    /// Command to the kicker
    pub kicker_cmd: KickerCmd,
}

impl PlayerCmd {
    pub fn zero(id: PlayerId) -> PlayerCmd {
        PlayerCmd {
            id,
            sx: 0.0,
            sy: 0.0,
            w: 0.0,
            dribble_speed: 0.0,
            kicker_cmd: KickerCmd::None,
        }
    }
}

impl Into<glue::Radio_Command> for PlayerCmd {
    fn into(self) -> glue::Radio_Command {
        glue::Radio_Command {
            speed: glue::HG_Pose {
                x: self.sx as f32,
                y: self.sy as f32,
                z: self.w as f32,
            },
            dribbler_speed: self.dribble_speed as f32,
            kicker_command: self.kicker_cmd.into(),
            _pad: [0, 0, 0],
            kick_time: 0.0,
            fan_speed: 0.0,
        }
    }
}

/// The status of a sub-system on the robot
#[derive(Debug, Copy, Clone)]
pub enum SysStatus {
    Emergency,
    Ok,
    Stop,
    Starting,
    Overtemp,
    NoReply,
    Armed,
    Disarmed,
    Safe,
    NotInstalled,
    Standby,
}

impl SysStatus {
    pub fn from_option(value: Option<glue::HG_Status>) -> Option<Self> {
        value.map(Into::into)
    }
}

impl Into<SysStatus> for glue::HG_Status {
    fn into(self) -> SysStatus {
        match self {
            Self::EMERGENCY => SysStatus::Emergency,
            Self::OK => SysStatus::Ok,
            Self::STOP => SysStatus::Stop,
            Self::STARTING => SysStatus::Starting,
            Self::OVERTEMP => SysStatus::Overtemp,
            Self::NO_REPLY => SysStatus::NoReply,
            Self::ARMED => SysStatus::Armed,
            Self::DISARMED => SysStatus::Disarmed,
            Self::SAFE => SysStatus::Safe,
            Self::NOT_INSTALLED => SysStatus::NotInstalled,
            Self::STANDBY => SysStatus::Standby,
        }
    }
}

/// A message from one of our robots to the AI
#[derive(Clone, Copy, Debug)]
pub struct PlayerFeedbackMsg {
    /// The robot's ID
    pub id: PlayerId,

    pub primary_status: Option<SysStatus>,
    pub kicker_status: Option<SysStatus>,
    pub imu_status: Option<SysStatus>,
    pub fan_status: Option<SysStatus>,
    pub kicker_cap_voltage: Option<f64>,
    pub kicker_temp: Option<f64>,
    pub motor_statuses: Option<[SysStatus; 5]>,
    pub motor_speeds: Option<[f64; 5]>,
    pub motor_temps: Option<[f64; 5]>,

    pub breakbeam_ball_detected: Option<bool>,
    pub breakbeam_sensor_ok: Option<bool>,
    pub pack_voltages: Option<[f64; 2]>,
}

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
