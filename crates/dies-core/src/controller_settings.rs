use serde::{Deserialize, Serialize};
use typeshare::typeshare;

/// Settings for the low-level controller.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct ControllerSettings {
    /// Maximum acceleration of the robot in mm/s².
    pub max_acceleration: f64,
    /// Maximum velocity of the robot in mm/s.
    pub max_velocity: f64,
    /// Maximum deceleration of the robot in mm/s².
    pub max_deceleration: f64,
    /// Maximum angular velocity of the robot in rad/s.
    pub max_angular_velocity: f64,
    /// Maximum angular acceleration of the robot in rad/s².
    pub max_angular_acceleration: f64,
    /// Maximum angular deceleration of the robot in rad/s².
    pub max_angular_deceleration: f64,
    /// Proportional gain for the close-range position controller.
    pub position_kp: f64,
    /// Time until destination in which the proportional controller is used, in seconds.
    pub position_proportional_time_window: f64,
    /// Proportional gain for the close-range angle controller.
    pub angle_kp: f64,
    /// Time until destination in which the proportional controller is used, in seconds.
    pub angle_proportional_time_window: f64,
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            max_acceleration: 700.0,
            max_velocity: 2000.0,
            max_deceleration: 900.0,
            max_angular_velocity: 30.0f64.to_radians(),
            max_angular_acceleration: 15.0f64.to_radians(),
            max_angular_deceleration: 15.0f64.to_radians(),
            position_kp: 0.7,
            position_proportional_time_window: 0.7,
            angle_kp: 1.0,
            angle_proportional_time_window: 0.02,
        }
    }
}
