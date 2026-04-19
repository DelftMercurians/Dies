use nalgebra::{Matrix2, Matrix4, Matrix4x2, Vector2, Vector4};

pub type Vec2 = Vector2<f64>;
pub type Mat2 = Matrix2<f64>;
pub type State = Vector4<f64>;
pub type Control = Vector2<f64>;
pub type StateJac = Matrix4<f64>;
pub type ControlJac = Matrix4x2<f64>;

pub const FWD: usize = 0;
pub const STRAFE: usize = 1;

/// Identifiable robot dynamics parameters, body-frame, shared across all robots.
///
/// The design doc specifies "9 globals"; the mass-normalized form here fuses
/// μ/m into a single `stiction` term per axis because only their ratio is
/// identifiable from velocity data, leaving 7 free scalars.
#[derive(Clone, Debug)]
pub struct RobotParams {
    /// Velocity-lag time constant per body axis [s]. Index: [FWD, STRAFE].
    pub tau: [f64; 2],
    /// Acceleration saturation per body axis [mm/s²].
    pub a_max: [f64; 2],
    /// Mass-normalized stiction magnitude per body axis [mm/s²].
    pub stiction: [f64; 2],
    /// Stiction smoothing scale [mm/s] — tanh transition width near zero velocity.
    pub v_eps: f64,
}

impl RobotParams {
    /// Hand-tuned default starting point for real SSL robots — useful before sysid has run.
    pub fn default_hand_tuned() -> Self {
        Self {
            tau: [0.08, 0.10],
            a_max: [6000.0, 4500.0],
            stiction: [150.0, 200.0],
            v_eps: 30.0,
        }
    }

    /// Pack into a 7-element array for sysid parameter vector operations.
    pub fn to_array(&self) -> [f64; 7] {
        [
            self.tau[FWD],
            self.tau[STRAFE],
            self.a_max[FWD],
            self.a_max[STRAFE],
            self.stiction[FWD],
            self.stiction[STRAFE],
            self.v_eps,
        ]
    }

    pub fn from_array(a: [f64; 7]) -> Self {
        Self {
            tau: [a[0], a[1]],
            a_max: [a[2], a[3]],
            stiction: [a[4], a[5]],
            v_eps: a[6],
        }
    }
}

/// Robot kinematic state — position and velocity in the global frame.
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

/// Obstacle geometry mirroring `dies_core::Obstacle` but living in this crate
/// to keep `dies-mpc` free of game-layer dependencies. The integration layer
/// translates at the boundary.
#[derive(Clone, Debug)]
pub enum ObstacleShape {
    Circle { center: Vec2, radius: f64 },
    Rectangle { min: Vec2, max: Vec2 },
    Line { start: Vec2, end: Vec2 },
}

/// Obstacle with linear future-motion prediction.
#[derive(Clone, Debug)]
pub struct PredictedObstacle {
    pub shape: ObstacleShape,
    /// Constant-velocity drift applied to the shape over the horizon [mm/s].
    pub velocity: Vec2,
    /// Surface-to-surface distance below which the penalty enters its high-cost region.
    pub safe_dist: f64,
    /// Surface-to-surface distance above which the barrier contributes zero cost.
    pub no_cost_dist: f64,
    /// Per-obstacle multiplier on top of the global obstacle weight. Used to
    /// make specific obstacles (goal areas, high-priority opponents) more
    /// strongly avoided than the baseline.
    pub weight_scale: f64,
}

/// Soft field-boundary penalty geometry.
#[derive(Clone, Debug)]
pub struct FieldBounds {
    pub half_length: f64,
    pub half_width: f64,
    pub penalty_depth: f64,
    pub penalty_half_width: f64,
    /// Inset where the soft wall begins ramping up [mm].
    pub margin: f64,
}

impl FieldBounds {
    pub fn centered(length: f64, width: f64, penalty_depth: f64, penalty_width: f64) -> Self {
        Self {
            half_length: 0.5 * length,
            half_width: 0.5 * width,
            penalty_depth,
            penalty_half_width: 0.5 * penalty_width,
            margin: 100.0,
        }
    }
}

/// All non-self world information MPC needs for a single solve. Goal-area
/// avoidance is the caller's responsibility: add goal-area rectangles to
/// `obstacles` with a higher `weight_scale`.
#[derive(Clone, Debug)]
pub struct WorldSnapshot {
    pub obstacles: Vec<PredictedObstacle>,
    pub field_bounds: FieldBounds,
}

/// Time-indexed reference sample (sorted by `t`).
#[derive(Clone, Debug)]
pub struct TimedRef {
    pub t: f64,
    pub pos: Vec2,
    pub vel: Option<Vec2>,
}

/// Reference trajectory the MPC tracks at each stage.
#[derive(Clone, Debug)]
pub enum ReferenceTrajectory {
    /// Constant target; no velocity reference.
    StaticPoint(Vec2),
    /// Time-indexed samples; queried via linear interpolation.
    Timed(Vec<TimedRef>),
}

/// How the terminal cost is constructed at stage N.
#[derive(Clone, Debug)]
pub enum TerminalMode {
    Position {
        p: Vec2,
    },
    PositionAndVelocity {
        p: Vec2,
        v: Vec2,
    },
    /// Match a moving target in both position and velocity — ball capture.
    RelativeVelocity {
        target_p: Vec2,
        target_v: Vec2,
    },
}

/// Cost weights. Magnitudes roughly expected: position/velocity/terminal in
/// 1e-4..1e-1 (positions in mm make the squared error huge), obstacle/field
/// around 1.
#[derive(Clone, Debug)]
pub struct CostWeights {
    pub position: f64,
    pub velocity: f64,
    pub control_smoothness: f64,
    pub terminal: f64,
    pub obstacle: f64,
    pub field_boundary: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        // Positions are mm, so squared-error tracking costs are numerically
        // huge; the small-looking position/terminal weights are intentional.
        // Obstacle and field-boundary weights are set to produce meaningful
        // avoidance at `care = 1.0` against typical tracking costs — `care`
        // then dials that up or down per-skill.
        Self {
            position: 1.0e-3,
            velocity: 5.0e-4,
            control_smoothness: 1.0e-5,
            terminal: 5.0e-2,
            obstacle: 200.0,
            field_boundary: 100.0,
        }
    }
}

/// Skill-facing MPC request.
#[derive(Clone, Debug)]
pub struct MpcTarget {
    pub reference: ReferenceTrajectory,
    pub terminal: TerminalMode,
    pub weights: CostWeights,
    /// Scales obstacle + field-boundary barriers. `[0, ∞)`; 1.0 = nominal.
    pub care: f64,
    /// Scales control-smoothness penalty. `[0, 1]`; 0 = full smoothness cost,
    /// 1 = no smoothness penalty (robot will gun it).
    pub aggressiveness: f64,
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

/// Solver tuning knobs. Defaults match the plan: 10-step horizon, 60 ms dt, 15 iters.
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

/// One timestamped observation used by sysid: what we commanded, what the
/// robot was actually doing, and the heading reported by its onboard IMU.
#[derive(Clone, Debug)]
pub struct Sample {
    pub t: f64,
    pub cmd: Vec2,
    pub heading: f64,
    pub state: RobotState,
}

#[derive(Clone, Debug)]
pub struct FitOptions {
    pub max_iters: u32,
    pub tol: f64,
    pub lambda_init: f64,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            max_iters: 100,
            tol: 1.0e-6,
            lambda_init: 1.0e-3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub params: RobotParams,
    pub residual_rms_per_axis: [f64; 2],
    pub iters: u32,
    pub converged: bool,
}
