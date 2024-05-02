mod geom;
mod world;

pub mod workspace_utils;

pub use geom::*;
pub use world::*;

pub type VisionMsg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
pub type GcRefereeMsg = dies_protos::ssl_gc_referee_message::Referee;

use serde::{Deserialize, Serialize};

/// A command to one of our players (placeholder)
///
/// All values are relative to the robots local frame and are in meters.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerCmd {
    /// The robot's ID
    pub id: u32,
    /// The player's x (left-right, with `+` left) velocity \[mm/s]
    pub sx: f32,
    /// The player's y (forward-backward, with `+` forward) velocity \[mm/s]
    pub sy: f32,
    /// The player's angular velocity (with `+` counter-clockwise, `-` clockwise) \[rad/s]
    pub w: f32,
    /// The player's dribble speed
    pub dribble_speed: f32,

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
}

/// A position command to one of our players
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlayerPosCmd {
    pub id: u32,
    pub x: f32,
    pub y: f32,
    pub orientation: f32,
}

/// A message from one of our robots to the AI
pub struct PlayerFeedbackMsg {
    /// The robot's ID
    pub id: u32,
    /// Capacitor voltage
    pub cap_v: f32,
}
