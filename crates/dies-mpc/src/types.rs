use nalgebra::{Matrix2, Matrix4, Matrix4x2, Vector2, Vector4};
use serde::{Deserialize, Serialize};

pub type Vec2 = Vector2<f64>;
pub type Mat2 = Matrix2<f64>;
pub type State = Vector4<f64>;
pub type Control = Vector2<f64>;
pub type StateJac = Matrix4<f64>;
pub type ControlJac = Matrix4x2<f64>;

pub const FWD: usize = 0;
pub const STRAFE: usize = 1;

/// Per-axis first-order velocity-lag time constants and acceleration ceilings.
///
/// Body-frame dynamics: `v̇_b[i] = a_max[i] · tanh((v_cmd_b[i] − v_b[i]) / (τ[i] · a_max[i]))`.
/// In the linear regime (small error) this collapses to the first-order lag
/// `(v_cmd − v) / τ`; far from steady-state the smooth saturation caps the
/// realised accel at ±a_max, modelling motor torque/current limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotParams {
    pub tau: [f64; 2],
    #[serde(default = "default_accel_max")]
    pub accel_max: [f64; 2],
}

fn default_accel_max() -> [f64; 2] {
    [3000.0, 3000.0]
}

impl RobotParams {
    pub fn default_hand_tuned() -> Self {
        Self {
            tau: [0.08, 0.10],
            accel_max: default_accel_max(),
        }
    }
}

/// Robot kinematic state — global-frame position and velocity.
#[derive(Clone, Copy, Debug)]
pub struct RobotState {
    pub pos: Vec2,
    pub vel: Vec2,
}

impl RobotState {
    pub fn to_state(&self) -> State {
        State::new(self.pos.x, self.pos.y, self.vel.x, self.vel.y)
    }

    pub fn from_state(s: &State) -> Self {
        Self {
            pos: Vec2::new(s[0], s[1]),
            vel: Vec2::new(s[2], s[3]),
        }
    }
}

/// Quadratic cost weights. All terms are `½ · w · ||residual||²`.
#[derive(Clone, Debug)]
pub struct CostWeights {
    /// Stage `||pos − target_p||²`. Pulls the trajectory toward the target.
    pub position: f64,
    /// Stage `||vel − target_v||²`. Pulls the trajectory toward the velocity reference.
    pub velocity: f64,
    /// Stage `||u||²`. Critical: keeps `Q_uu` non-degenerate so iLQR feedback
    /// gains don't blow up when every other term has zero curvature in `u`.
    pub control: f64,
    /// Stage `||u − u_prev||²`. Damps high-frequency control oscillation.
    pub control_smoothness: f64,
    /// Terminal `||pos_N − target_p||²`.
    pub terminal_position: f64,
    /// Terminal `||vel_N − target_v||²`. Encourages stopping at the target.
    pub terminal_velocity: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            // Positions are mm so squared errors are huge; keep weights small.
            position: 1.0e-3,
            velocity: 5.0e-4,
            // Floor on Q_uu eigenvalues — prevents the runaway feedback gain
            // we hit with no `||u||²` penalty (commanded ±500 mm/s at the
            // target with vision noise as input).
            control: 5.0e-4,
            control_smoothness: 1.0e-4,
            terminal_position: 5.0e-2,
            terminal_velocity: 5.0e-2,
        }
    }
}

/// Skill-facing MPC request. Single static target; velocity reference defaults
/// to zero (i.e. "go to p and stop").
#[derive(Clone, Debug)]
pub struct MpcTarget {
    pub p: Vec2,
    pub v: Vec2,
    pub weights: CostWeights,
}

impl MpcTarget {
    /// Convenience: go to `p` and come to rest, default weights.
    pub fn goto(p: Vec2) -> Self {
        Self {
            p,
            v: Vec2::zeros(),
            weights: CostWeights::default(),
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
            horizon: 10,
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
