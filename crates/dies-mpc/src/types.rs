use nalgebra::{Matrix2, Matrix5, Matrix5x3, Vector2, Vector3, Vector5};

use crate::obstacle::Obstacle;

// iLQR tuning parameters live in `dies-core` so they can sit on
// `ExecutorSettings` and flow through the standard settings pipeline. Re-export
// them here so solver-side paths (`crate::types::RobotParams`, …) keep working.
pub use dies_core::{CostWeights, ObstacleConfig, RobotParams};

pub type Vec2 = Vector2<f64>;
pub type Mat2 = Matrix2<f64>;
/// Optimised state `[px, py, vx, vy, theta]` (global frame; heading in rad).
pub type State = Vector5<f64>;
/// Control `[vx_cmd, vy_cmd, theta_cmd]` (global frame; heading setpoint in rad).
pub type Control = Vector3<f64>;
pub type StateJac = Matrix5<f64>;
pub type ControlJac = Matrix5x3<f64>;

pub const FWD: usize = 0;
pub const STRAFE: usize = 1;

/// Robot kinematic state — global-frame position, velocity, and heading.
#[derive(Clone, Copy, Debug)]
pub struct RobotState {
    pub pos: Vec2,
    pub vel: Vec2,
    /// Current heading in radians (global frame).
    pub heading: f64,
}

impl RobotState {
    pub fn to_state(&self) -> State {
        State::new(self.pos.x, self.pos.y, self.vel.x, self.vel.y, self.heading)
    }

    pub fn from_state(s: &State) -> Self {
        Self {
            pos: Vec2::new(s[0], s[1]),
            vel: Vec2::new(s[2], s[3]),
            heading: s[4],
        }
    }
}

/// Skill-facing MPC request. Single static target; velocity reference defaults
/// to zero (i.e. "go to p and stop").
#[derive(Clone, Debug)]
pub struct MpcTarget {
    pub p: Vec2,
    pub v: Vec2,
    /// Desired heading in radians (global frame). Only attracts the trajectory
    /// when `weights.heading > 0`; set the weight to zero to leave heading free
    /// so the planner picks whatever orientation best serves translation.
    pub heading: f64,
    pub weights: CostWeights,
    /// Soft obstacles active for this solve, in the solver's (team-relative)
    /// frame. Empty by default — the bare tracking problem.
    pub obstacles: Vec<Obstacle>,
}

impl MpcTarget {
    /// Convenience: go to `p` and come to rest, default weights, no obstacles,
    /// heading free.
    pub fn goto(p: Vec2) -> Self {
        Self {
            p,
            v: Vec2::zeros(),
            heading: 0.0,
            weights: CostWeights::default(),
            obstacles: Vec::new(),
        }
    }
}

/// Trajectory as a pair of state (N+1) and control (N) sequences.
#[derive(Clone, Debug)]
pub struct Trajectory {
    pub states: Vec<State>,
    pub controls: Vec<Control>,
}

impl Trajectory {
    pub fn new(horizon: usize) -> Self {
        Self {
            states: vec![State::zeros(); horizon + 1],
            controls: vec![Control::zeros(); horizon],
        }
    }
}

#[derive(Clone, Debug)]
pub struct SolveResult {
    pub trajectory: Trajectory,
    pub final_cost: f64,
    pub iters: u32,
    pub converged: bool,
    pub solve_time_us: u64,
}

/// Solver tuning knobs.
#[derive(Clone, Debug)]
pub struct SolverConfig {
    pub horizon: usize,
    pub dt: f64,
    pub max_iters: u32,
    pub cost_tol: f64,
    pub reg_init: f64,
    pub reg_min: f64,
    pub reg_max: f64,
    pub reg_factor: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            horizon: 100,
            dt: 0.06,
            max_iters: 15,
            cost_tol: 1.0e-3,
            reg_init: 1.0e-6,
            reg_min: 1.0e-8,
            reg_max: 1.0e3,
            reg_factor: 4.0,
        }
    }
}
