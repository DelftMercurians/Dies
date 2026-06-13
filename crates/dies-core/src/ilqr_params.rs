//! iLQR / MPC tuning parameters.
//!
//! These live in `dies-core` (rather than in `dies-mpc`, where the solver that
//! consumes them lives) so they can sit on [`crate::ExecutorSettings`] and ride
//! the standard settings pipeline: webui edit → live update → JSON persist.
//! `dies-mpc` re-exports them from its `types` module, so solver-side paths
//! (`crate::types::RobotParams`, …) are unaffected.

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

/// Per-axis first-order velocity-lag time constants and acceleration ceilings.
///
/// Body-frame dynamics: `v̇_b[i] = a_max[i] · tanh((v_cmd_b[i] − v_b[i]) / (τ[i] · a_max[i]))`.
/// In the linear regime (small error) this collapses to the first-order lag
/// `(v_cmd − v) / τ`; far from steady-state the smooth saturation caps the
/// realised accel at ±a_max, modelling motor torque/current limits.
///
/// Index `0` is the forward (FWD) body axis, `1` is the strafe (STRAFE) axis.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct RobotParams {
    pub tau: [f64; 2],
    #[serde(default = "default_accel_max")]
    pub accel_max: [f64; 2],
    /// First-order heading-lag time constant [s]. Models the onboard IMU yaw
    /// loop slewing toward the commanded heading setpoint.
    #[serde(default = "default_tau_yaw")]
    pub tau_yaw: f64,
    /// Heading slew-rate ceiling [rad/s] — the tanh saturation on `thetȧ`.
    #[serde(default = "default_omega_max")]
    pub omega_max: f64,
    /// Soft obstacle-avoidance knobs. Defaulted so older params files (which
    /// only carried `tau` / `accel_max`) keep parsing.
    #[serde(default)]
    pub obstacles: ObstacleConfig,
    /// Quadratic tracking-cost weights. Defaulted so older params files keep
    /// parsing; applied to every solve via `MpcTarget`.
    #[serde(default)]
    pub weights: CostWeights,
}

fn default_accel_max() -> [f64; 2] {
    [3500.0, 3500.0]
}

fn default_tau_yaw() -> f64 {
    0.1
}

fn default_omega_max() -> f64 {
    20.0
}

impl RobotParams {
    pub fn default_hand_tuned() -> Self {
        Self {
            tau: [0.08, 0.10],
            accel_max: default_accel_max(),
            tau_yaw: default_tau_yaw(),
            omega_max: default_omega_max(),
            obstacles: ObstacleConfig::default(),
            weights: CostWeights::default(),
        }
    }
}

impl Default for RobotParams {
    fn default() -> Self {
        Self::default_hand_tuned()
    }
}

/// Tunable parameters for the soft obstacle barriers the integration layer
/// builds around each robot (robots, field walls, defense areas). A single
/// `weight` / `influence` pair is shared across obstacle types; per-type
/// geometry margins are kept separate.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct ObstacleConfig {
    /// Barrier stiffness `w` shared by every obstacle term.
    pub weight: f64,
    /// Influence distance `δ` [mm]: how far out the barrier starts to push.
    pub influence: f64,
    /// Extra clearance added on top of the two-robot contact distance [mm].
    pub robot_clearance: f64,
    /// How far ahead to extrapolate moving robots at constant velocity [s].
    pub robot_extrapolation: f64,
    /// Inset of the robot centre from the physical field walls [mm].
    pub wall_margin: f64,
    /// Keep-out margin grown around each defense area [mm].
    pub defense_margin: f64,
}

impl Default for ObstacleConfig {
    fn default() -> Self {
        Self {
            weight: 0.1,
            influence: 300.0,
            robot_clearance: 50.0,
            robot_extrapolation: 0.5,
            wall_margin: 120.0,
            defense_margin: 50.0,
        }
    }
}

/// Quadratic cost weights. All terms are `½ · w · ||residual||²`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[typeshare]
pub struct CostWeights {
    /// Stage `||pos − target_p||²`. Pulls the trajectory toward the target.
    #[serde(default = "default_weight_position")]
    pub position: f64,
    /// Stage `||vel − target_v||²`. Pulls the trajectory toward the velocity
    /// reference (zero by default → "arrive and stop"). The main anti-overshoot
    /// knob: nonzero values damp arrival velocity.
    #[serde(default = "default_weight_velocity")]
    pub velocity: f64,
    /// Stage `||u||²`. Critical: keeps `Q_uu` non-degenerate so iLQR feedback
    /// gains don't blow up when every other term has zero curvature in `u`.
    #[serde(default = "default_weight_control")]
    pub control: f64,
    /// Stage `||u − u_prev||²` (translational controls only). Damps
    /// high-frequency control oscillation.
    #[serde(default = "default_weight_control_smoothness")]
    pub control_smoothness: f64,
    /// Stage `1 − cos(theta − theta_d)`. Attracts the robot heading toward the
    /// target heading. Zero ⇒ heading is left free for the planner to optimise
    /// purely in service of translation.
    #[serde(default = "default_weight_heading")]
    pub heading: f64,
    /// Stage `½·(theta_cmd − theta)²`. Regularises the heading setpoint (the
    /// turn effort) and keeps `Q_uu` non-degenerate in the `theta_cmd` axis —
    /// the heading analogue of the `control` term.
    #[serde(default = "default_weight_heading_control")]
    pub heading_control: f64,
}

// Positions are mm so squared errors are huge; keep weights small.
fn default_weight_position() -> f64 {
    1.0e-3
}
fn default_weight_velocity() -> f64 {
    0.0
}
fn default_weight_control() -> f64 {
    5.0e-5
}
fn default_weight_control_smoothness() -> f64 {
    1.0e-4
}
// Heading residuals are O(1) rad, so these are O(1) unlike the mm-scale
// translational weights.
fn default_weight_heading() -> f64 {
    10.0
}
fn default_weight_heading_control() -> f64 {
    1.0
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            position: default_weight_position(),
            velocity: default_weight_velocity(),
            control: default_weight_control(),
            control_smoothness: default_weight_control_smoothness(),
            heading: default_weight_heading(),
            heading_control: default_weight_heading_control(),
        }
    }
}
