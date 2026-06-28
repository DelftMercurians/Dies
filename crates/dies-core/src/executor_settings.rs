use std::{collections::HashMap, fs, path::Path};

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::{
    avoidance_config::AvoidanceConfig, skill_settings::SkillSettings, FieldGeometry, PlayerId,
    PossessionConfig, SideAssignment,
};

/// Settings for the low-level controller.
///
/// Every field here is consumed by `PlayerController` or one of its
/// subcontrollers (`PathFollower`, `YawController`).
/// Low-level controller limits and path-follower tuning. The translational path
/// follower (`PathFollower`) produces a speed profile (cruise / cornering /
/// braking-to-goal) and the player controller applies a first-order asymmetric
/// acceleration clamp; `YawController` handles heading.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[typeshare]
pub struct ControllerSettings {
    // --- Translational limits.
    /// Acceleration cap when speeding up (mm/s²).
    pub max_acceleration: f64,
    /// Deceleration cap when slowing down (mm/s²); also the braking authority the
    /// follower's speed profile plans against.
    pub max_deceleration: f64,
    /// Speed cap on the commanded velocity magnitude (mm/s).
    pub max_velocity: f64,
    /// Lateral (cornering) acceleration cap (mm/s²) — sets how fast the robot may
    /// carry through a path corner.
    pub lateral_acceleration: f64,
    /// Proportional arrival gain (1/s): commanded speed eases off as `kp ×
    /// remaining distance` into corners and the goal. Over-damped (no overshoot);
    /// lower = gentler/earlier braking, higher = later/snappier.
    pub approach_kp: f64,
    /// Active-braking gain for the terminal approach. When the robot is
    /// overspeeding relative to the proportional arrival profile, the commanded
    /// velocity is pushed *below* the profile (and may reverse) by `brake_gain ×
    /// overspeed`, and the acceleration slew clamp is bypassed so the firmware
    /// reverse-thrusts to a hard stop. `0` = disabled (gentle proportional only);
    /// `1` roughly mirrors the overspeed into a reverse command. Per-robot
    /// overridable via `PlayerControlInput::brake_gain`.
    #[serde(default)]
    pub brake_gain: f64,

    // --- Pure-pursuit path following.
    /// Minimum pure-pursuit lookahead distance (mm), used at low speed.
    pub lookahead_min: f64,
    /// Pure-pursuit lookahead time (s): lookahead = clamp(time·speed, min, max).
    pub lookahead_time: f64,

    // --- Rotational limits (used by YawController).
    /// Maximum angular velocity (rad/s).
    pub max_angular_velocity: f64,
    /// Maximum angular acceleration (rad/s²) — passed to YawController as the
    /// upper bound on its planned angular accel.
    pub max_angular_acceleration: f64,

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
            max_deceleration: 6000.0,
            max_velocity: 3000.0,
            lateral_acceleration: 4000.0,
            approach_kp: 4.0,
            brake_gain: 0.0,
            lookahead_min: 200.0,
            lookahead_time: 0.2,
            max_angular_velocity: 25.132741228718345, // 8π rad/s
            max_angular_acceleration: 349.0658503988659, // ~20π rad/s²
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
#[serde(default)]
#[typeshare]
pub struct TrackerSettings {
    /// Vision crop applied at the ball tracker — fractions of the field
    /// half-extent. Defaults to the full field.
    pub field_mask: FieldMask,

    /// Constant-velocity process noise (white-noise *acceleration* PSD). Used when
    /// `player_use_acceleration` is false.
    pub player_unit_transition_var: f64,
    /// Constant-acceleration process noise (white-noise *jerk* PSD). Used when
    /// `player_use_acceleration` is true. On a totally different numeric scale from
    /// the CV value, which is why it is a separate field — toggling the model must
    /// not silently leave the wrong process-noise magnitude in place.
    #[serde(default = "default_player_ca_var")]
    pub player_ca_unit_transition_var: f64,
    /// Measurement noise for the player position Kalman filter (mm²). Should match
    /// the real vision noise floor (~0.4 mm² measured on hardware).
    pub player_measurement_var: f64,
    /// Use a constant-acceleration motion model (state x,vx,ax,…) for players
    /// instead of constant-velocity. The CA model anticipates sustained
    /// acceleration, cutting the onset/transient velocity lag in hard maneuvers.
    #[serde(default = "default_player_use_acceleration")]
    pub player_use_acceleration: bool,
    /// Feed the commanded velocity into the filter's predict step as a first-order
    /// lag toward the setpoint (feedforward). Reduces lag further during commanded
    /// maneuvers. EXPERIMENTAL — needs field tuning of `player_command_tau`.
    #[serde(default)]
    pub player_use_command_feedforward: bool,
    /// First-order time constant (s) for command feedforward: the predicted velocity
    /// relaxes toward the commanded velocity with this τ. ~robot velocity-loop
    /// response time. Only used when `player_use_command_feedforward` is true.
    #[serde(default = "default_player_command_tau")]
    pub player_command_tau: f64,
    /// EWMA factor for the player yaw low-pass filter — 0 = no filtering,
    /// 1 = freeze.
    pub player_yaw_lpf_alpha: f64,

    /// Process noise for the ball position/velocity Kalman filter (mm²/s).
    pub ball_unit_transition_var: f64,
    /// Measurement noise for the ball position Kalman filter (mm²).
    pub ball_measurement_var: f64,
    /// Minimum vision confidence for a ball detection to be accepted. Some camera
    /// setups report near-zero confidence even for a clean ball, so this defaults
    /// low — raise it only if your vision reports trustworthy confidence values.
    #[serde(default = "default_ball_confidence_threshold")]
    pub ball_confidence_threshold: f64,
}

fn default_ball_confidence_threshold() -> f64 {
    0.0
}

fn default_player_ca_var() -> f64 {
    1.0e6
}

fn default_player_use_acceleration() -> bool {
    true
}

fn default_player_command_tau() -> f64 {
    0.15
}

impl Default for TrackerSettings {
    fn default() -> Self {
        Self {
            // Tuned against 2026-06-13 logs (replay-harness Pareto): R≈measured
            // vision noise (0.4 mm²). CA model on by default — it eliminates the
            // hard-maneuver onset lag; CV values retained for toggling back.
            player_unit_transition_var: 3000.0,
            player_ca_unit_transition_var: default_player_ca_var(),
            player_measurement_var: 0.4,
            player_use_acceleration: true,
            player_use_command_feedforward: false,
            player_command_tau: default_player_command_tau(),
            player_yaw_lpf_alpha: 0.15,
            ball_unit_transition_var: 20.48,
            ball_measurement_var: 0.01,
            ball_confidence_threshold: default_ball_confidence_threshold(),
            field_mask: FieldMask::default(),
        }
    }
}

/// Team configuration that can be set before executor creation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
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
#[serde(default)]
#[typeshare]
pub struct TeamSpecificSettings {
    pub handicaps: HashMap<PlayerId, Vec<Handicap>>,
}

/// Settings for the executor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[typeshare]
pub struct ExecutorSettings {
    pub controller_settings: ControllerSettings,
    pub tracker_settings: TrackerSettings,
    pub team_configuration: TeamConfiguration,
    pub yellow_team_settings: TeamSpecificSettings,
    pub blue_team_settings: TeamSpecificSettings,
    pub skill_settings: SkillSettings,
    /// Tuning for the unified ball-possession metric (computed in the world tracker).
    #[serde(default)]
    pub possession_config: PossessionConfig,
    pub allow_no_vision: bool,
    /// Global on/off for goal-area avoidance. When false, both compliance and
    /// the controller skip the goal-area keep-out logic (goalkeeper exception
    /// is irrelevant since the whole feature is off).
    #[serde(default = "default_true")]
    pub goal_area_avoidance: bool,
    /// Collision-avoidance tuning (obstacle margins + global planner + ORCA).
    /// Applied live to the planner and ORCA solver.
    #[serde(default)]
    pub avoidance: AvoidanceConfig,
    /// Dev-only: when set, the strategy host watches each strategy binary on
    /// disk and hot-swaps the process when it is rebuilt. Runtime flag set by
    /// the CLI (`--strategy-mode watch`); never persisted.
    #[serde(skip)]
    pub hot_reload: bool,
    #[serde(skip)]
    pub vision_delay_ms: u32,
    /// Headless self-play only: run the strategy IPC in blocking lockstep mode
    /// (host waits for each reply every tick) so matches are deterministic.
    /// Runtime flag set by the CLI `self-play` command; never persisted.
    #[serde(skip)]
    pub strategy_blocking: bool,
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
                    let msg = format!(
                        "Failed to parse {}: {err}. FALLING BACK TO ALL-DEFAULT SETTINGS \
                         (your tuned values are NOT in effect). Fix the file and restart.",
                        path.as_ref().display()
                    );
                    eprintln!("{msg}");
                    log::error!("{msg}");
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
            possession_config: PossessionConfig::default(),
            allow_no_vision: false,
            goal_area_avoidance: true,
            avoidance: AvoidanceConfig::default(),
            hot_reload: false,
            vision_delay_ms: 0,
            strategy_blocking: false,
        }
    }
}
