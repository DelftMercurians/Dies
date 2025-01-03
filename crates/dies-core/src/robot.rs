use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RobotCmd {
    Move(RobotMoveCmd),
    SetHeadingReference { heading: f64 },
}

/// A command to one of our players as it will be sent to the robot.
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct RobotMoveCmd {
    /// The player's x (forward-backward, with `+` forward) velocity \[m/s]
    pub sx: f64,
    /// The player's y (left-right, with `+` right) velocity \[m/s]
    pub sy: f64,
    /// The player's angular velocity (with `+` counter-clockwise, `-` clockwise) \[rad/s]
    pub w: f64,
    /// The player's dribble speed
    pub dribble_speed: f64,
    pub fan_speed: f64,
    pub kick_speed: f64,
    pub mainboard_cmd: RobotMainboardCmd,
}

/// A command directly to the mainboard of a robot.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
pub enum RobotMainboardCmd {
    /// Do nothing
    #[default]
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
    /// Reboot the robot
    Reboot,
    /// Beep the buzzer
    Beep,
    /// Coast the motors
    Coast,
    /// Control the heading of the robot
    HeadingControl,
    /// Control the yaw rate of the robot
    YawRateControl,
}

/// A message from one of our robots to the AI
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RobotFeedback {
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

/// The status of a sub-system on the robot
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
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
