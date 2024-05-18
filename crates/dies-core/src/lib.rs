mod geom;
mod world;

pub mod workspace_utils;

pub use geom::*;
pub use world::*;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

use serde::{Deserialize, Serialize};

/// A command to one of our players as it will be sent to the robot.
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerCmd {
    /// The robot's ID
    pub id: u32,
    /// The player's x (left-right, with `+` left) velocity \[m/s]
    pub sx: f64,
    /// The player's y (forward-backward, with `+` forward) velocity \[m/s]
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
    pub fn zero(id: u32) -> PlayerCmd {
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

/// A position command to one of our players
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayerPosCmd {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub orientation: f64,
}

/// A message from one of our robots to the AI
pub struct PlayerFeedbackMsg {
    /// The robot's ID
    pub id: u32,
    /// Capacitor voltage
    pub cap_v: f64,
}
