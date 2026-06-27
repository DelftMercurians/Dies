//! Per-player control overrides.
//!
//! A [`ControlOverride`] is a sparse diff over the executor's default control
//! parameters for a single robot. Every field is optional; `None` means "leave
//! the executor default in place". Strategies attach one per frame via
//! `PlayerHandle::set_control_override`; the executor merges the `Some(..)`
//! fields onto the `PlayerControlInput` it builds for that robot.
//!
//! Unlike skill commands, these are plain scalars/bools and carry no spatial
//! meaning, so they cross the IPC boundary without any coordinate transform.

use serde::{Deserialize, Serialize};

/// Sparse per-robot overrides for the position controller and avoidance stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ControlOverride {
    /// Single snappiness dial. Scales the position approach gain (`approach_kp`)
    /// and the terminal active-braking gain together. `0.0` reproduces the
    /// global defaults; higher is more aggressive. `None` = global default.
    pub aggressiveness: Option<f64>,
    /// Explicit terminal active-braking gain, decoupled from `aggressiveness`.
    /// Takes precedence over the `aggressiveness`-derived brake gain.
    pub brake_gain: Option<f64>,
    /// Per-robot speed cap (mm/s).
    pub speed_limit: Option<f64>,
    /// Per-robot acceleration cap (mm/s²).
    pub accel_limit: Option<f64>,
    /// `Some(false)` disables ORCA for this robot — its commanded velocity is
    /// never deflected by reciprocal avoidance or static (field-boundary /
    /// defense-area) constraints.
    pub avoid_robots: Option<bool>,
    /// `Some(false)` disables the global path planner for this robot — it drives
    /// straight at its target instead of routing around obstacles/walls.
    pub use_planner: Option<bool>,
}

impl ControlOverride {
    /// An empty override (all defaults).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the aggressiveness dial (scales position gain + braking).
    pub fn aggressiveness(mut self, v: f64) -> Self {
        self.aggressiveness = Some(v);
        self
    }

    /// Set an explicit terminal active-braking gain.
    pub fn brake_gain(mut self, v: f64) -> Self {
        self.brake_gain = Some(v);
        self
    }

    /// Set a per-robot speed cap (mm/s).
    pub fn speed_limit(mut self, v: f64) -> Self {
        self.speed_limit = Some(v);
        self
    }

    /// Set a per-robot acceleration cap (mm/s²).
    pub fn accel_limit(mut self, v: f64) -> Self {
        self.accel_limit = Some(v);
        self
    }

    /// Enable/disable ORCA for this robot. `false` removes all velocity
    /// deflection, including field-boundary avoidance.
    pub fn avoid_robots(mut self, v: bool) -> Self {
        self.avoid_robots = Some(v);
        self
    }

    /// Enable/disable the global path planner for this robot.
    pub fn use_planner(mut self, v: bool) -> Self {
        self.use_planner = Some(v);
        self
    }
}
