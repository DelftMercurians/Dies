//! Collision-avoidance tuning parameters.
//!
//! Lives in `dies-core` (rather than in `dies-executor`, where the avoidance
//! modules that consume it live) so it can sit on [`crate::ExecutorSettings`]
//! and ride the standard settings pipeline: webui edit → live update → JSON
//! persist. Consumed by the executor's `control::avoidance` modules: the
//! obstacle-set builder, the global path planner, and the ORCA solver.

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

/// Tuning for the two-layer collision-avoidance stack (global planner + ORCA)
/// and the shared obstacle model they both read.
///
/// All distances are millimetres, times are seconds. A single `AvoidanceConfig`
/// is shared across the whole team and applied live from settings updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
#[typeshare]
pub struct AvoidanceConfig {
    // --- Obstacle margins ---
    /// ORCA's emergency margin on top of the two-robot contact distance [mm].
    /// ORCA is a last-resort reactive backstop, so keep this small (≈0) — the
    /// planner does the real avoidance with `planner_margin`. The planner inflates
    /// robots by `robot_clearance + planner_margin`; ORCA only by this.
    pub robot_clearance: f64,
    /// Inset of the robot centre from the physical field walls [mm].
    pub wall_margin: f64,
    /// Keep-out margin grown around each defense area [mm].
    pub defense_margin: f64,
    /// Ball keep-out radius during `Stop` [mm] (rule: stay 500 mm clear).
    pub ball_stop_radius: f64,
    /// Base ball keep-out radius when `avoid_ball` is set [mm].
    pub ball_base_radius: f64,
    /// Additional ball keep-out radius scaled by `avoid_ball_care` [mm].
    pub ball_care_scale: f64,
    /// How far ahead moving robots are extrapolated at constant velocity [s].
    pub robot_extrapolation: f64,

    // --- ORCA (reactive emergency backstop only) ---
    /// ORCA time horizon τ [s]: how far ahead reciprocal collisions are
    /// resolved. Kept short so ORCA only fires for imminent collisions, not in
    /// normal play (planned avoidance handles the rest).
    pub time_horizon: f64,
    /// Speed below which a robot counts as "stationary" for reciprocity gating
    /// [mm/s]. A stationary robot does not yield to a moving one — the mover
    /// takes the full avoidance burden.
    pub stationary_speed: f64,
    /// Prefer steering around obstacles over braking for them: drop ORCA's
    /// cut-off-circle projection so a constrained velocity is deflected (turned)
    /// rather than slowed. Avoids the "sticky crawl" where ORCA suppresses speed
    /// near an obstacle. Falls back to the 3-D LP when genuinely boxed in.
    pub prefer_steering: bool,
    /// Only robots within this distance are considered ORCA neighbours [mm].
    pub neighbor_dist: f64,
    /// Cap on the number of neighbours folded into a single ORCA LP.
    #[typeshare(serialized_as = "u32")]
    pub max_neighbors: usize,

    // --- Global planner ---
    /// Grid cell size for the planner's any-angle search [mm].
    pub grid_resolution: f64,
    /// The planner's robot-avoidance margin [mm]: it routes paths clear of
    /// robots by `robot_clearance + planner_margin`. This is the primary
    /// avoidance — keep it comfortable so ORCA rarely needs to engage.
    pub planner_margin: f64,
    /// Replan only when the target has moved more than this [mm] (hysteresis to
    /// suppress path flicker).
    pub replan_target_tol: f64,

    // --- Master toggles ---
    /// Run the global path planner. When false, the follower steers straight at
    /// the target and only ORCA provides avoidance.
    pub planner_enabled: bool,
    /// Run ORCA reciprocal avoidance. When false, the path-follower velocity
    /// passes through untouched (useful for isolating layers when debugging).
    pub orca_enabled: bool,
}

impl Default for AvoidanceConfig {
    fn default() -> Self {
        Self {
            // ORCA emergency backstop: ~no extra margin, short horizon → fires
            // only for imminent collisions, not in normal play.
            robot_clearance: 0.0,
            time_horizon: 0.5,
            wall_margin: 120.0,
            defense_margin: 50.0,
            ball_stop_radius: 800.0,
            ball_base_radius: 100.0,
            ball_care_scale: 100.0,
            robot_extrapolation: 0.5,
            stationary_speed: 50.0,
            prefer_steering: false,
            neighbor_dist: 2000.0,
            max_neighbors: 8,
            grid_resolution: 100.0,
            // Primary (planned) avoidance margin around robots.
            planner_margin: 150.0,
            replan_target_tol: 150.0,
            planner_enabled: true,
            orca_enabled: true,
        }
    }
}
