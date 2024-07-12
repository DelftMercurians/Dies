use std::{fs, path::Path};

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
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub position_cutoff_distance: f64,
    /// Proportional gain for the close-range angle controller.
    pub angle_kp: f64,
    /// Time until destination in which the proportional controller is used, in seconds.
    pub angle_proportional_time_window: f64,
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub angle_cutoff_distance: f64,

    /// Attractive force coefficient for players
    pub force_alpha: f64,
    /// Repulsive force coefficient for players
    pub force_beta: f64,
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
            position_cutoff_distance: 0.,
            angle_kp: 1.0,
            angle_proportional_time_window: 0.02,
            angle_cutoff_distance: 0.,
            force_alpha: 0.001,
            force_beta: 0.01,
        }
    }
}

/// Settings for the `WorldTracker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct TrackerSettings {
    /// Whether our team color is blue
    pub is_blue: bool,
    /// The initial sign of the enemy goal's x coordinate in ssl-vision coordinates.
    pub initial_opp_goal_x: f64,

    /// Transition variance for the player Kalman filter.
    pub player_unit_transition_var: f64,
    /// Measurement variance for the player Kalman filter.
    pub player_measurement_var: f64,
    /// Smoothinfg factor for the yaw LPF
    pub player_yaw_lpf_alpha: f64,

    /// Transition variance for the ball Kalman filter.
    pub ball_unit_transition_var: f64,
    /// Measurement variance for the ball Kalman filter.
    pub ball_measurement_var: f64,
}

impl Default for TrackerSettings {
    fn default() -> Self {
        Self {
            is_blue: true,
            initial_opp_goal_x: 1.0,
            player_unit_transition_var: 0.1,
            player_measurement_var: 2.0,
            ball_unit_transition_var: 10.0,
            ball_measurement_var: 10.0,
            player_yaw_lpf_alpha: 0.5,
        }
    }
}

/// Settings for the executor.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[typeshare]
pub struct ExecutorSettings {
    pub controller_settings: ControllerSettings,
    pub tracker_settings: TrackerSettings,
}

impl ExecutorSettings {
    /// Load the executor settings from a file, or store the default settings if the file does not
    /// exist or is invalid.
    ///
    /// # Panics
    ///
    /// Panics if the file exists but cannot be read or if creating the file fails.
    pub fn load_or_insert(path: impl AsRef<Path>) -> Self {
        match fs::read_to_string(path.as_ref()) {
            Ok(contents) => match serde_json::from_str(&contents) {
                Ok(settings) => settings,
                Err(err) => {
                    eprintln!("Failed to parse executor settings: {}", err);
                    Self::default()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let settings = Self::default();
                fs::write(path, serde_json::to_string_pretty(&settings).unwrap())
                    .expect("Failed to write executor settings");
                settings
            }
            Err(err) => panic!("Failed to read executor settings: {}", err),
        }
    }

    /// Store the executor settings in the given file.
    pub async fn store(&self, path: impl AsRef<Path>) {
        if let Err(err) = tokio::fs::write(path, serde_json::to_string_pretty(self).unwrap()).await
        {
            log::error!("Failed to write executor settings: {}", err);
        }
    }
}
