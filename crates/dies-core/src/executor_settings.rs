use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{FieldGeometry, PlayerId, SideAssignment};

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
    /// Proportional gain for the close-range position controller.
    pub position_kp: f64,
    /// Time until destination in which the proportional controller is used, in seconds.
    pub position_proportional_time_window: f64,
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub position_cutoff_distance: f64,
    /// Proportional gain for the close-range angle controller.
    pub angle_kp: f64,
    /// Distance used as threshold for the controller to prevent shaky behavior
    pub angle_cutoff_distance: f64,
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            max_acceleration: 125000.0,
            max_velocity: 3120.0,
            max_deceleration: 6340.0,
            max_angular_velocity: 25.132741228718345,
            max_angular_acceleration: 349.0658503988659,
            position_kp: 50000.0,
            position_proportional_time_window: 1.1,
            position_cutoff_distance: 70.0,
            angle_kp: 2.8,
            angle_cutoff_distance: 0.03490658503988659,
        }
    }
}

/// A field mask for the `WorldTracker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct FieldMask {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl FieldMask {
    pub fn contains(&self, x: f32, y: f32, field_geom: Option<&FieldGeometry>) -> bool {
        if let Some(field_geom) = field_geom {
            let x = x as f64;
            let y = y as f64;
            let x_min = (field_geom.field_length / 2.0 + field_geom.boundary_width) * self.x_min;
            let x_max = (field_geom.field_length / 2.0 + field_geom.boundary_width) * self.x_max;
            let y_min = (field_geom.field_width / 2.0 + field_geom.boundary_width) * self.y_min;
            let y_max = (field_geom.field_width / 2.0 + field_geom.boundary_width) * self.y_max;
            x >= x_min && x <= x_max && y >= y_min && y <= y_max
        } else {
            false
        }
    }
}

impl Default for FieldMask {
    fn default() -> Self {
        Self {
            x_min: -1.0,
            x_max: 1.0,
            y_min: -1.0,
            y_max: 1.0,
        }
    }
}

/// Settings for the `WorldTracker`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct TrackerSettings {
    pub field_mask: FieldMask,

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
            player_unit_transition_var: 95.75,
            player_measurement_var: 0.01,
            player_yaw_lpf_alpha: 0.15,
            ball_unit_transition_var: 20.48,
            ball_measurement_var: 0.01,
            field_mask: FieldMask::default(),
        }
    }
}

/// Team configuration that can be set before executor creation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct TeamConfiguration {
    /// Whether the blue team is active
    pub blue_active: bool,
    /// Whether the yellow team is active
    pub yellow_active: bool,
    /// Script path for the blue team
    pub blue_script_path: Option<String>,
    /// Script path for the yellow team
    pub yellow_script_path: Option<String>,
    /// Which team defends the positive x side
    pub side_assignment: SideAssignment,
}

impl Default for TeamConfiguration {
    fn default() -> Self {
        Self {
            blue_active: true,
            yellow_active: false,
            blue_script_path: None,
            yellow_script_path: None,
            side_assignment: SideAssignment::YellowOnPositive,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum Handicap {
    NoKicker,
    NoDribbler,
}

impl std::fmt::Display for Handicap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Handicap::NoKicker => write!(f, "No kicker"),
            Handicap::NoDribbler => write!(f, "No dribbler"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[typeshare]
pub struct TeamSpecificSettings {
    pub handicaps: HashMap<PlayerId, Vec<Handicap>>,
}

/// Settings for the executor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct ExecutorSettings {
    pub controller_settings: ControllerSettings,
    pub tracker_settings: TrackerSettings,
    pub team_configuration: TeamConfiguration,
    pub yellow_team_settings: TeamSpecificSettings,
    pub blue_team_settings: TeamSpecificSettings,
    pub allow_no_vision: bool,
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

impl Default for ExecutorSettings {
    fn default() -> Self {
        Self {
            controller_settings: ControllerSettings::default(),
            tracker_settings: TrackerSettings::default(),
            team_configuration: TeamConfiguration::default(),
            yellow_team_settings: TeamSpecificSettings::default(),
            blue_team_settings: TeamSpecificSettings::default(),
            allow_no_vision: true,
        }
    }
}
