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
    /// Arm the kicker (smart kick: fires when the kick counter increments)
    Arm,
    /// Arm a reflex kick: the firmware fires automatically when the ball is
    /// detected at the breakbeam. No counter increment needed — holding this
    /// mode keeps the reflex armed (packet-loss safe).
    ArmReflex,
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
    /// Calibrate the IMU (bench diagnostics)
    CalibrateImu,
    /// Calibrate the breakbeam sensor (bench diagnostics)
    CalibrateBreakbeam,
}

impl From<RobotCmd> for glue::Radio_RobotCommand {
    fn from(val: RobotCmd) -> Self {
        match val {
            RobotCmd::None => glue::Radio_RobotCommand::NONE,
            RobotCmd::Arm => glue::Radio_RobotCommand::ARM_COUNTER_KICK,
            RobotCmd::ArmReflex => glue::Radio_RobotCommand::ARM_REFLEX_KICK,
            RobotCmd::Disarm => glue::Radio_RobotCommand::DISARM,
            RobotCmd::Discharge => glue::Radio_RobotCommand::DISCHARGE,
            RobotCmd::Kick => glue::Radio_RobotCommand::KICK,
            RobotCmd::Chip => glue::Radio_RobotCommand::CHIP,
            RobotCmd::PowerBoardOff => glue::Radio_RobotCommand::POWER_BOARD_OFF,
            RobotCmd::Reboot => glue::Radio_RobotCommand::REBOOT,
            RobotCmd::Beep => glue::Radio_RobotCommand::BEEP,
            RobotCmd::Coast => glue::Radio_RobotCommand::COAST,
            RobotCmd::CalibrateImu => glue::Radio_RobotCommand::CALIBRATE_IMU,
            RobotCmd::CalibrateBreakbeam => glue::Radio_RobotCommand::CALIBRATE_BB,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum PlayerCmd {
    Move(PlayerMoveCmd),
    GlobalMove(PlayerGlobalMoveCmd),
}

impl PlayerCmd {
    pub fn id(&self) -> PlayerId {
        match self {
            PlayerCmd::Move(cmd) => cmd.id,
            PlayerCmd::GlobalMove(cmd) => cmd.id,
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
            gen_command: glue::Radio_GenericCommand {
                dribbler_speed_i: val.dribble_speed as i16,
                kick_time_i: (5_000.0 * val.kick_speed as f32) as u16,
                time_to_kick: 0,
                smart_kick_couter: 0,
                robot_command: val.robot_cmd.into(),
            },
            _pad: [0, 0, 0, 0, 0, 0, 0, 0],
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum RotationDirection {
    Clockwise,
    CounterClockwise,
    NoPreference,
}

impl RotationDirection {
    pub fn as_i8(self) -> i8 {
        match self {
            RotationDirection::Clockwise => -1,
            RotationDirection::CounterClockwise => 1,
            RotationDirection::NoPreference => 0,
        }
    }

    pub fn from_f64(val: f64) -> Self {
        match val {
            -1.0 => RotationDirection::Clockwise,
            1.0 => RotationDirection::CounterClockwise,
            _ => RotationDirection::NoPreference,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayerGlobalMoveCmd {
    pub id: PlayerId,
    pub global_x: f64,
    pub global_y: f64,
    pub heading_setpoint: f64,
    pub last_heading: f64,
    pub dribble_speed: f64,
    pub kick_counter: u8,
    pub robot_cmd: RobotCmd,
    pub w: f64,
    /// Maximum yaw rate in rad/s
    pub max_yaw_rate: f64,
    pub preferred_rotation_direction: RotationDirection,
}

impl PlayerGlobalMoveCmd {
    pub fn zero(id: PlayerId) -> PlayerGlobalMoveCmd {
        PlayerGlobalMoveCmd {
            id,
            global_x: 0.0,
            global_y: 0.0,
            heading_setpoint: f64::NAN,
            last_heading: 0.0,
            dribble_speed: 0.0,
            kick_counter: 0,
            robot_cmd: RobotCmd::None,
            w: 0.0,
            max_yaw_rate: 0.0,
            preferred_rotation_direction: RotationDirection::NoPreference,
        }
    }
}

impl From<PlayerGlobalMoveCmd> for glue::Radio_GlobalCommand {
    fn from(val: PlayerGlobalMoveCmd) -> Self {
        glue::Radio_GlobalCommand {
            global_speed_x: val.global_x as f32,
            global_speed_y: val.global_y as f32,
            heading_last_measurement: val.last_heading as f32,
            heading_setpoint: val.heading_setpoint as f32,
            gen_command: glue::Radio_GenericCommand {
                dribbler_speed_i: val.dribble_speed as i16,
                kick_time_i: 5_000,
                time_to_kick: 0,
                smart_kick_couter: val.kick_counter,
                robot_command: val.robot_cmd.into(),
            },
            max_yaw_rate: (val.max_yaw_rate * 10.0) as u16,
            preferred_rotation_direction: val.preferred_rotation_direction.as_i8(),
            _pad: 0,
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
        yaw: Option<Angle>,
        /// Dribbler speed normalised to \[0, 1\]
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in local frame
    LocalVelocity {
        velocity: Vector2,
        yaw: Option<Angle>,
        /// Dribbler speed normalised to \[0, 1\]
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in global frame
    GlobalVelocity {
        velocity: Vector2,
        yaw: Option<Angle>,
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
    Cooldown,
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
            glue::HG_Status::COOLDOWN => SysStatus::Cooldown,
        }
    }
}

/// State of the firmware reflex-kick state machine.
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[typeshare]
pub enum ReflexKickState {
    Off,
    Armed,
    Cooldown,
    Emergency,
}

impl From<glue::HG_ReflexState> for ReflexKickState {
    fn from(val: glue::HG_ReflexState) -> Self {
        match val {
            glue::HG_ReflexState::OFF => ReflexKickState::Off,
            glue::HG_ReflexState::ARMED => ReflexKickState::Armed,
            glue::HG_ReflexState::COOLDOWN => ReflexKickState::Cooldown,
            glue::HG_ReflexState::EMERGENCY => ReflexKickState::Emergency,
        }
    }
}

/// Echo of the last command the robot reports having received (global frame).
#[derive(Debug, Copy, Clone, Serialize)]
#[typeshare]
pub struct CommandEcho {
    pub global_speed_x: f32,
    pub global_speed_y: f32,
    pub heading_setpoint: f32,
    pub heading_last_measurement: f32,
    pub dribbler_speed: i16,
    pub kick_time: u16,
    /// Raw `Radio_RobotCommand` code; the frontend maps it to a label.
    pub robot_command: u8,
}

/// Firmware version reported by a robot.
#[derive(Debug, Copy, Clone, Serialize)]
#[typeshare]
pub struct FirmwareVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
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
    pub tof_status: Option<SysStatus>,
    pub kicker_cap_voltage: Option<f32>,
    pub kicker_temp: Option<f32>,
    pub motor_statuses: Option<[SysStatus; 5]>,
    pub motor_speeds: Option<[f32; 5]>,
    pub motor_temps: Option<[f32; 5]>,

    pub breakbeam_ball_detected: Option<bool>,
    pub breakbeam_sensor_ok: Option<bool>,
    pub pack_voltages: Option<[f32; 2]>,

    /// IMU readings: [ang_x, ang_y, ang_z, ang_wx, ang_wy, ang_wz]
    pub imu_readings: Option<[f32; 6]>,

    // --- Extended bench-dashboard telemetry ---
    pub motor_currents: Option<[f32; 5]>,
    pub main_board_current: Option<f32>,
    pub avg_loop_time_us: Option<u32>,
    pub max_loop_time_us: Option<u32>,
    pub smart_kick_counter: Option<u8>,
    pub kick_ok_flag: Option<bool>,
    pub reflex_kick_state: Option<ReflexKickState>,
    pub reflex_kick_counter: Option<u8>,
    pub breakbeam_raw: Option<u16>,
    pub tof_ball_detected: Option<bool>,
    /// Time-of-flight ball position estimate `[x, y]`.
    pub tof_xy: Option<[i32; 2]>,
    pub last_command: Option<CommandEcho>,
    pub firmware_version: Option<FirmwareVersion>,
}

impl PlayerFeedbackMsg {
    pub fn empty(id: PlayerId) -> Self {
        Self {
            id,
            primary_status: None,
            kicker_status: None,
            imu_status: None,
            tof_status: None,
            kicker_cap_voltage: None,
            kicker_temp: None,
            motor_statuses: None,
            motor_speeds: None,
            motor_temps: None,
            breakbeam_ball_detected: None,
            breakbeam_sensor_ok: None,
            pack_voltages: None,
            imu_readings: None,
            motor_currents: None,
            main_board_current: None,
            avg_loop_time_us: None,
            max_loop_time_us: None,
            smart_kick_counter: None,
            kick_ok_flag: None,
            reflex_kick_state: None,
            reflex_kick_counter: None,
            breakbeam_raw: None,
            tof_ball_detected: None,
            tof_xy: None,
            last_command: None,
            firmware_version: None,
        }
    }
}

/// Status of the RF basestation itself, surfaced to the bench UI.
#[derive(Clone, Debug, Serialize)]
#[typeshare]
pub struct BaseStationInfo {
    pub connected: bool,
    pub protocol_ok: bool,
    pub version: String,
    pub protocol_version: String,
    /// Current radio channel in MHz.
    pub channel_mhz: u32,
    pub num_radios: u8,
    /// Online flag per radio module.
    pub radios_online: Vec<bool>,
    pub max_robots: u8,
}

/// Motion frame for a bench `SetMotion` command.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[typeshare]
pub enum BenchMotionMode {
    /// `(vx, vy)` in the robot's local frame, `w_or_heading` is `w` \[rad/s].
    Local,
    /// `(vx, vy)` in the global frame, `w_or_heading` is the heading setpoint \[rad].
    Global,
}

/// A one-shot bench action targeting a single robot (or broadcast).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum BenchOneShot {
    Beep,
    Reboot,
    Shutdown,
    Coast,
    Arm,
    Disarm,
    Discharge,
    Kick { speed: f64 },
    ArmReflex,
    CalibrateBreakbeam,
    CalibrateImu,
    GetVersion,
    ZeroHeading,
    SetHeading { angle: f64 },
}

/// A command from the bench UI, sent straight to the basestation (no executor,
/// no vision). Robots are addressed by raw robot id.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub enum BenchCommand {
    /// Continuously-streamed motion setpoint for a taken robot.
    SetMotion {
        robot_id: u32,
        mode: BenchMotionMode,
        vx: f64,
        vy: f64,
        /// `w` \[rad/s] in Local mode, heading setpoint \[rad] in Global mode.
        w_or_heading: f64,
        dribble_speed: f64,
    },
    /// Zero a single robot's setpoint.
    Stop { robot_id: u32 },
    /// Zero every robot and release control.
    StopAll,
    /// Take or release streaming control of a robot.
    TakeControl { robot_id: u32, taken: bool },
    /// Fire a one-shot action at a single robot.
    OneShot { robot_id: u32, kind: BenchOneShot },
    /// Broadcast a one-shot action to all robots.
    Broadcast { kind: BenchOneShot },
    /// Set the radio channel of the base (`robot_id = None`) or a robot.
    SetChannel { robot_id: Option<u32>, channel: u8 },
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
    Waller,
}
