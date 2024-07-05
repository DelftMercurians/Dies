use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::Angle;

use super::Vector2;

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

    pub fn into_proto_v0_with_id(self, with_id: usize) -> String {
        let extra = match self.kicker_cmd {
            KickerCmd::Arm => "A".to_string(),
            KickerCmd::Disarm => "D".to_string(),
            KickerCmd::Kick => "K".to_string(),
            KickerCmd::Discharge => "".to_string(),
            KickerCmd::None => "".to_string(),
            KickerCmd::Chip => "".to_string(),
            KickerCmd::PowerBoardOff => "".to_string(),
        };

        format!(
            "p{};Sx{:.2};Sy{:.2};Sz{:.2};Sd{:.0};Kt7000;S.{};\n",
            with_id, self.sx, self.sy, self.w, self.dribble_speed, extra
        )
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

/// An override command for a player for manual control.
#[typeshare]
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(tag = "type", content = "data")]
pub enum PlayerOverrideCommand {
    /// Do nothing
    #[default]
    Stop,
    /// Move the robot to a globel position and yaw
    MoveTo {
        position: Vector2,
        yaw: Angle,
        /// Dribbler speed normalised to \[0, 1\]
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in local frame
    LocalVelocity {
        velocity: Vector2,
        angular_velocity: f64,
        /// Dribbler speed normalised to \[0, 1\]
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in global frame
    GlobalVelocity {
        velocity: Vector2,
        angular_velocity: f64,
        /// Dribbler speed normalised to \[0, 1\]
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Engage the kicker
    Kick { speed: f64 },
    /// Discharge the kicker safely
    DischargeKicker,
}

impl PlayerOverrideCommand {
    pub fn dribble_speed(&self) -> f64 {
        match self {
            PlayerOverrideCommand::MoveTo { dribble_speed, .. } => *dribble_speed,
            PlayerOverrideCommand::LocalVelocity { dribble_speed, .. } => *dribble_speed,
            PlayerOverrideCommand::GlobalVelocity { dribble_speed, .. } => *dribble_speed,
            PlayerOverrideCommand::Stop => 0.0,
            PlayerOverrideCommand::Kick { .. } => 0.0,
            PlayerOverrideCommand::DischargeKicker => 0.0,
        }
    }

    pub fn arm_kick(&self) -> bool {
        match self {
            PlayerOverrideCommand::MoveTo { arm_kick, .. } => *arm_kick,
            PlayerOverrideCommand::LocalVelocity { arm_kick, .. } => *arm_kick,
            PlayerOverrideCommand::GlobalVelocity { arm_kick, .. } => *arm_kick,
            PlayerOverrideCommand::Stop => false,
            PlayerOverrideCommand::Kick { .. } => false,
            PlayerOverrideCommand::DischargeKicker => false,
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
    pub kicker_cap_voltage: Option<f32>,
    pub kicker_temp: Option<f32>,
    pub motor_statuses: Option<[SysStatus; 5]>,
    pub motor_speeds: Option<[f32; 5]>,
    pub motor_temps: Option<[f32; 5]>,

    pub breakbeam_ball_detected: Option<bool>,
    pub breakbeam_sensor_ok: Option<bool>,
    pub pack_voltages: Option<[f32; 2]>,
}
