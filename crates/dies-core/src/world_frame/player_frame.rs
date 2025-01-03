use serde::{Deserialize, Serialize};

use crate::{Angle, PlayerId, SysStatus, Vector2};

/// A struct to store the player state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlayerFrame {
    /// The player's unique id
    pub id: PlayerId,
    /// Position of the player filtered by us in mm
    pub position: Vector2,
    /// Velocity of the player in mm/s
    pub velocity: Vector2,
    /// Yaw of the player, in radians, (-pi, pi)
    pub yaw: Angle,
    /// Angular speed of the player (in rad/s)
    pub angular_speed: f64,
    /// Feedback from the player, if it is controlled
    pub feedback: PlayerFeedback,
}

impl PlayerFrame {
    pub fn new(id: PlayerId) -> Self {
        Self {
            id,
            position: Vector2::zeros(),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }
    }
}

/// The feedback that the player receives from the world, if it is controlled.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PlayerFeedback {
    NotControlled,
    NotReceived,
    Controlled {
        /// The overall status of the robot.
        primary_status: Option<SysStatus>,
        /// The voltage of the kicker capacitor (in V).
        kicker_cap_voltage: Option<f32>,
        /// The temperature of the kicker.
        kicker_temp: Option<f32>,
        /// The voltages of the battery packs.
        pack_voltages: Option<[f32; 2]>,
        /// Whether the breakbeam sensor detected a ball.
        breakbeam_ball_detected: bool,
        /// The status of the IMU.
        imu_status: Option<SysStatus>,
        /// The status of the kicker.
        kicker_status: Option<SysStatus>,
    },
}
