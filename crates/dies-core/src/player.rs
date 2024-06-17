use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::Angle;

use super::Vector2;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash, PartialEq, Eq)]
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

/// A command to one of our players as it will be sent to the robot.
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug)]
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

    pub arm: bool,
    pub disarm: bool,
    pub kick: bool,
}

impl PlayerCmd {
    pub fn zero(id: PlayerId) -> PlayerCmd {
        PlayerCmd {
            id,
            sx: 0.0,
            sy: 0.0,
            w: 0.0,
            dribble_speed: 0.0,
            arm: false,
            disarm: false,
            kick: false,
        }
    }

    pub fn to_string(&self) -> String {
        let extra = if self.disarm {
            "D".to_string()
        } else if self.arm {
            "A".to_string()
        } else if self.kick {
            "K".to_string()
        } else {
            "".to_string()
        };

        format!(
            "p{};Sx{:.2};Sy{:.2};Sz{:.2};Sd{:.0};Kt7000;S.{};\n",
            self.id, self.sx, self.sy, self.w, self.dribble_speed, extra
        )
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
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in local frame
    LocalVelocity {
        velocity: Vector2,
        angular_velocity: f64,
        dribble_speed: f64,
        arm_kick: bool,
    },
    /// Move the robot with velocity in global frame
    GlobalVelocity {
        velocity: Vector2,
        angular_velocity: f64,
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
