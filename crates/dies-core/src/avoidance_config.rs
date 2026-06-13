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
#[typeshare]
pub struct AvoidanceConfig {
    // --- Obstacle margins (shared by planner + ORCA) ---
    /// Extra clearance added on top of the two-robot contact distance [mm].
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

    // --- ORCA ---
    /// ORCA time horizon τ [s]: how far ahead reciprocal collisions are
    /// resolved. Larger ⇒ ORCA reacts earlier but brakes harder near obstacles;
    /// smaller ⇒ later but smoother. The planner margin (below) keeps planned
    /// paths outside ORCA's braking zone so this can stay modest.
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
    /// Extra clearance the planner leaves around robots **on top of** ORCA's
    /// hard combined radius [mm], so planned paths sit outside the band where
    /// ORCA actively brakes and the two layers don't fight each other.
    pub planner_margin: f64,
    /// Distance at which the robot is considered to have reached an intermediate
    /// waypoint and advances to the next [mm]. Wider than the final-target
    /// cutoff so the robot flows through corners (pass-through) instead of
    /// braking to each one. The final target still decelerates normally.
    pub waypoint_tolerance: f64,
    /// Replan only when the target has moved more than this [mm] (hysteresis to
    /// suppress path flicker).
    pub replan_target_tol: f64,

    // --- Master toggles (replace the old ControllerMode enum) ---
    /// Run the global path planner. When false, MTP steers straight at the
    /// target and only ORCA provides avoidance.
    pub planner_enabled: bool,
    /// Run ORCA reciprocal avoidance. When false, the MTP velocity passes
    /// through untouched (useful for isolating layers when debugging).
    pub orca_enabled: bool,
}

impl Default for AvoidanceConfig {
    fn default() -> Self {
        Self {
            robot_clearance: 50.0,
            wall_margin: 120.0,
            defense_margin: 50.0,
            ball_stop_radius: 800.0,
            ball_base_radius: 100.0,
            ball_care_scale: 100.0,
            robot_extrapolation: 0.5,
            time_horizon: 1.0,
            stationary_speed: 50.0,
            // Off by default: removing the cut-off projection makes ORCA unable
            // to slow-and-commit when a target sits directly behind a close
            // obstacle, so it can orbit/stall. Opt in to trade that for never
            // crawling. The jerk-limited tracker already smooths the cut-off's
            // slowdowns, so leaving this off is usually the better default.
            prefer_steering: false,
            neighbor_dist: 2000.0,
            max_neighbors: 8,
            grid_resolution: 100.0,
            planner_margin: 100.0,
            waypoint_tolerance: 200.0,
            replan_target_tol: 150.0,
            planner_enabled: true,
            orca_enabled: true,
        }
    }
}
