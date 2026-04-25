use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{skill_settings::SkillSettings, FieldGeometry, PlayerId, SideAssignment};

/// Settings for the low-level controller.
///
/// Every field here is consumed by `PlayerController` or one of its
/// subcontrollers (TwoStepMTP, YawController). The hard accel clamp in
/// `PlayerController::command` reads `max_acceleration` directly and
/// constrains the per-tick Δv for *both* MTP and iLQR outputs.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct ControllerSettings {
    // --- Translational limits (hard caps applied at the player-controller
    // boundary; iLQR/MTP both go through them).
    /// Per-axis acceleration cap for translational motion (mm/s²).
    pub max_acceleration: f64,
    /// Speed cap on the commanded velocity magnitude (mm/s).
    pub max_velocity: f64,
    /// Deceleration cap used when MTP plans braking (mm/s²).
    pub max_deceleration: f64,

    // --- Rotational limits (used by YawController).
    /// Maximum angular velocity (rad/s).
    pub max_angular_velocity: f64,
    /// Maximum angular acceleration (rad/s²) — passed to YawController as the
    /// upper bound on its planned angular accel.
    pub max_angular_acceleration: f64,

    // --- TwoStepMTP gains.
    /// Proportional gain for translational tracking inside the MTP cutoff.
    pub position_kp: f64,
    /// Position deadzone (mm) — MTP commands zero velocity inside this radius.
    pub position_cutoff_distance: f64,

    // --- YawController gains.
    /// Proportional gain for heading tracking.
    pub angle_kp: f64,
    /// Heading deadzone (rad) — yaw control outputs zero inside this band.
    pub angle_cutoff_distance: f64,
}

impl Default for ControllerSettings {
    fn default() -> Self {
        Self {
            max_acceleration: 4000.0,
            max_velocity: 3000.0,
            max_deceleration: 6000.0,
            max_angular_velocity: 25.132741228718345, // 8π rad/s
            max_angular_acceleration: 349.0658503988659, // ~20π rad/s²
            position_kp: 2.0,
            position_cutoff_distance: 15.0,
            angle_kp: 2.8,
            angle_cutoff_distance: 0.03490658503988659, // 2°
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
///
/// Kalman variances are unit-time process noise (`*_unit_transition_var`,
/// in mm²/s for the constant-velocity model) and measurement variance
/// (`*_measurement_var`, in mm²). Higher transition variance → tracker
/// trusts measurements more (snappier, noisier). Higher measurement
/// variance → tracker trusts dynamics model more (smoother, laggier).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct TrackerSettings {
    /// Vision crop applied at the ball tracker — fractions of the field
    /// half-extent. Defaults to the full field.
    pub field_mask: FieldMask,

    /// Process noise for the player position/velocity Kalman filter (mm²/s).
    pub player_unit_transition_var: f64,
    /// Measurement noise for the player position Kalman filter (mm²).
    pub player_measurement_var: f64,
    /// EWMA factor for the player yaw low-pass filter — 0 = no filtering,
    /// 1 = freeze.
    pub player_yaw_lpf_alpha: f64,

    /// Process noise for the ball position/velocity Kalman filter (mm²/s).
    pub ball_unit_transition_var: f64,
    /// Measurement noise for the ball position Kalman filter (mm²).
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
    /// Which team defends the positive x side
    pub side_assignment: SideAssignment,
    /// IPC strategy binary name for blue team (None = no strategy)
    pub blue_strategy: Option<String>,
    /// IPC strategy binary name for yellow team (None = no strategy)
    pub yellow_strategy: Option<String>,
}

impl Default for TeamConfiguration {
    fn default() -> Self {
        Self {
            blue_active: true,
            yellow_active: false,
            side_assignment: SideAssignment::YellowOnPositive,
            blue_strategy: None,
            yellow_strategy: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum Handicap {
    NoKicker,
    NoDribbler,
    NoBreakbeam,
}

impl std::fmt::Display for Handicap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Handicap::NoKicker => write!(f, "No kicker"),
            Handicap::NoDribbler => write!(f, "No dribbler"),
            Handicap::NoBreakbeam => write!(f, "No breakbeam"),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[typeshare]
pub struct TeamSpecificSettings {
    pub handicaps: HashMap<PlayerId, Vec<Handicap>>,
}

/// Which low-level motion controller the executor runs. Global across all
/// robots; switched at runtime through the webui settings panel or at
/// startup via `dies-cli --controller`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[typeshare]
pub enum ControllerMode {
    /// Two-step minimum-time-path controller (default, battle-tested).
    #[default]
    Mtp,
    /// iLQR MPC from `dies-mpc`.
    Ilqr,
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
    pub skill_settings: SkillSettings,
    pub allow_no_vision: bool,
    #[serde(default)]
    pub controller_mode: ControllerMode,
    /// Global on/off for goal-area avoidance. When false, both compliance and
    /// the controller skip the goal-area keep-out logic (goalkeeper exception
    /// is irrelevant since the whole feature is off).
    #[serde(default = "default_true")]
    pub goal_area_avoidance: bool,
}

fn default_true() -> bool {
    true
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
            skill_settings: SkillSettings::default(),
            allow_no_vision: true,
            controller_mode: ControllerMode::default(),
            goal_area_avoidance: true,
        }
    }
}
