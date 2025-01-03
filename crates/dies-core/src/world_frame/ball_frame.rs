use serde::{Deserialize, Serialize};

use crate::Vector3;

/// A struct to store the ball state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BallFrame {
    /// Position of the ball filtered by us, in mm, in dies coordinates
    pub position: Vector3,
    /// Velocity of the ball in mm/s, in dies coordinates
    pub velocity: Vector3,
    /// Whether the ball is being detected
    pub detected: bool,
}
