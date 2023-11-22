mod env;
mod geom;
mod runtime;
mod world;

pub mod workspace_utils;

pub use env::*;
pub use geom::*;
pub use runtime::*;
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
}
