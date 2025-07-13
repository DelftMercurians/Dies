use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use super::Vector2;
use crate::Angle;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[typeshare(serialized_as = "u32")]
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
pub enum RobotCmd {
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
    Reboot,
    Beep,
    Coast,
    HeadingControl,
    YawRateControl,
}

impl From<RobotCmd> for glue::Radio_RobotCommand {
    fn from(val: RobotCmd) -> Self {
        match val {
            RobotCmd::None => glue::Radio_RobotCommand::NONE,
            RobotCmd::Arm => glue::Radio_RobotCommand::ARM,
            RobotCmd::Disarm => glue::Radio_RobotCommand::DISARM,
            RobotCmd::Discharge => glue::Radio_RobotCommand::DISCHARGE,
            RobotCmd::Kick => glue::Radio_RobotCommand::KICK,
            RobotCmd::Chip => glue::Radio_RobotCommand::CHIP,
            RobotCmd::PowerBoardOff => glue::Radio_RobotCommand::POWER_BOARD_OFF,
            RobotCmd::Reboot => glue::Radio_RobotCommand::REBOOT,
            RobotCmd::Beep => glue::Radio_RobotCommand::BEEP,
            RobotCmd::Coast => glue::Radio_RobotCommand::COAST,
            RobotCmd::HeadingControl => glue::Radio_RobotCommand::HEADING_CONTROL,
            RobotCmd::YawRateControl => glue::Radio_RobotCommand::YAW_RATE_CONTROL,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PlayerCmd {
    Move(PlayerMoveCmd),
}

impl PlayerCmd {
    pub fn id(&self) -> PlayerId {
        match self {
            PlayerCmd::Move(cmd) => cmd.id,
        }
    }
}

/// A command to one of our players as it will be sent to the robot.
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayerMoveCmd {
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
    pub robot_cmd: RobotCmd,
    pub fan_speed: f64,
    pub kick_speed: f64,
}

impl PlayerMoveCmd {
    pub fn zero(id: PlayerId) -> PlayerMoveCmd {
        PlayerMoveCmd {
            id,
            sx: 0.0,
            sy: 0.0,
            w: 0.0,
            dribble_speed: 0.0,
            robot_cmd: RobotCmd::None,
            fan_speed: 0.0,
            kick_speed: 0.0,
        }
    }

    // pub fn is_zero(&self) -> bool {
    //     self.sx == 0.0 && self.sy == 0.0 && self.w == 0.0 && self.dribble_speed == 0.0 && self.robot_cmd
    // }

    pub fn into_proto_v0_with_id(self, with_id: usize) -> String {
        let extra = match self.robot_cmd {
            RobotCmd::Arm => "A".to_string(),
            RobotCmd::Disarm => "D".to_string(),
            RobotCmd::Kick => "K".to_string(),
            RobotCmd::Discharge => "".to_string(),
            RobotCmd::None => "".to_string(),
            RobotCmd::Chip => "".to_string(),
            RobotCmd::PowerBoardOff => "".to_string(),
            _ => "".to_string(),
        };

        format!(
            "p{};Sx{:.2};Sy{:.2};Sz{:.2};Sd{:.0};Kt7000;S.{};\n",
            with_id, self.sx, self.sy, self.w, self.dribble_speed, extra
        )
    }
}

impl From<PlayerMoveCmd> for glue::Radio_Command {
    fn from(val: PlayerMoveCmd) -> Self {
        glue::Radio_Command {
            speed: glue::HG_Pose {
                x: val.sx as f32,
                y: val.sy as f32,
                z: val.w as f32,
            },
            dribbler_speed: val.dribble_speed as f32,
            robot_command: val.robot_cmd.into(),
            kick_time: 5_000.0,
            fan_speed: 0.0, //val.fan_speed as f32,
            _pad: [0, 0, 0],
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
    Kick {
        speed: f64,
    },
    /// Discharge the kicker safely
    DischargeKicker,

    SetFanSpeed {
        speed: f64,
    },
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
            PlayerOverrideCommand::SetFanSpeed { .. } => 0.0,
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
            PlayerOverrideCommand::SetFanSpeed { .. } => false,
        }
    }
}

/// The status of a sub-system on the robot
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[typeshare]
pub enum SysStatus {
    Emergency,
    Ok,
    Ready,
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

impl From<glue::HG_Status> for SysStatus {
    fn from(val: glue::HG_Status) -> Self {
        match val {
            glue::HG_Status::EMERGENCY => SysStatus::Emergency,
            glue::HG_Status::READY => SysStatus::Ready,
            glue::HG_Status::OK => SysStatus::Ok,
            glue::HG_Status::STOP => SysStatus::Stop,
            glue::HG_Status::STARTING => SysStatus::Starting,
            glue::HG_Status::OVERTEMP => SysStatus::Overtemp,
            glue::HG_Status::NO_REPLY => SysStatus::NoReply,
            glue::HG_Status::ARMED => SysStatus::Armed,
            glue::HG_Status::DISARMED => SysStatus::Disarmed,
            glue::HG_Status::SAFE => SysStatus::Safe,
            glue::HG_Status::NOT_INSTALLED => SysStatus::NotInstalled,
            glue::HG_Status::STANDBY => SysStatus::Standby,
        }
    }
}

/// A message from one of our robots to the AI
#[derive(Clone, Copy, Debug, Serialize)]
#[typeshare]
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

impl PlayerFeedbackMsg {
    pub fn empty(id: PlayerId) -> Self {
        Self {
            id,
            primary_status: None,
            kicker_status: None,
            imu_status: None,
            fan_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            motor_statuses: None,
            motor_speeds: None,
            motor_temps: None,
            breakbeam_ball_detected: None,
            breakbeam_sensor_ok: None,
            pack_voltages: None,
        }
    }
}

/// Role of a player according to the game rules. These are mainly for rule-compliance.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[typeshare]
pub enum RoleType {
    /// A regular player with no special role
    #[default]
    Player,
    /// The goalkeeper
    Goalkeeper,
    /// The attacking kicker during kick-off
    KickoffKicker,
    /// penalty kicker
    PenaltyKicker,
    /// freekicker
    FreeKicker,
}
