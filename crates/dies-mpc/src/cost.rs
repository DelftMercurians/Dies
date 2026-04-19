//! Parameterized stage + terminal cost used by the iLQR solver.
//!
//! Every term is expressed as one or more scalar residuals `r_i(x, u)` with a
//! quadratic cost contribution `½ · r_i²` — a natural fit for the
//! Gauss-Newton Hessian approximation `∇²L ≈ Jᵀ J` that iLQR uses. Because
//! the residual factor also serves as the barrier return, obstacle and
//! field-boundary penalties plug in here uniformly.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, RowVector2, RowVector4, Vector2, Vector4};

use crate::barrier::{halfplane_residual, obstacle_residual};
use crate::types::{
    MpcTarget, ObstacleShape, PredictedObstacle, ReferenceTrajectory, TerminalMode, Vec2,
    WorldSnapshot,
};

#[derive(Clone, Debug)]
pub struct StageDerivs {
    pub cost: f64,
    pub lx: Vector4<f64>,
    pub lu: Vector2<f64>,
    pub lxx: Matrix4<f64>,
    pub luu: Matrix2<f64>,
    /// `∂²L / (∂u ∂x)` with shape 2×4.
    pub lux: Matrix2x4<f64>,
}

impl StageDerivs {
    fn zeros() -> Self {
        Self {
            cost: 0.0,
            lx: Vector4::zeros(),
            lu: Vector2::zeros(),
            lxx: Matrix4::zeros(),
            luu: Matrix2::zeros(),
            lux: Matrix2x4::zeros(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct TerminalDerivs {
    pub cost: f64,
    pub lx: Vector4<f64>,
    pub lxx: Matrix4<f64>,
}

impl TerminalDerivs {
    fn zeros() -> Self {
        Self {
            cost: 0.0,
            lx: Vector4::zeros(),
            lxx: Matrix4::zeros(),
        }
    }
}

/// Interpolate a reference trajectory at time `t`. Returns
/// `(position_reference, optional_velocity_reference)`.
pub fn eval_reference(r: &ReferenceTrajectory, t: f64) -> (Vec2, Option<Vec2>) {
    match r {
        ReferenceTrajectory::StaticPoint(p) => (*p, None),
        ReferenceTrajectory::Timed(samples) => {
            if samples.is_empty() {
                return (Vec2::zeros(), None);
            }
            if t <= samples[0].t {
                return (samples[0].pos, samples[0].vel);
            }
            let last = samples.last().unwrap();
            if t >= last.t {
                return (last.pos, last.vel);
            }
            // Binary search for the bracketing interval.
            let idx = samples
                .partition_point(|s| s.t <= t)
                .saturating_sub(1)
                .min(samples.len() - 2);
            let s0 = &samples[idx];
            let s1 = &samples[idx + 1];
            let denom = (s1.t - s0.t).max(1.0e-9);
            let alpha = ((t - s0.t) / denom).clamp(0.0, 1.0);
            let pos = s0.pos + (s1.pos - s0.pos) * alpha;
            let vel = match (s0.vel, s1.vel) {
                (Some(v0), Some(v1)) => Some(v0 + (v1 - v0) * alpha),
                _ => None,
            };
            (pos, vel)
        }
    }
}

/// Translate an obstacle shape by its constant-velocity drift over `t` seconds.
fn predict_shape(shape: &ObstacleShape, vel: Vec2, t: f64) -> ObstacleShape {
    let d = vel * t;
    match shape {
        ObstacleShape::Circle { center, radius } => ObstacleShape::Circle {
            center: center + d,
            radius: *radius,
        },
        ObstacleShape::Rectangle { min, max } => ObstacleShape::Rectangle {
            min: min + d,
            max: max + d,
        },
        ObstacleShape::Line { start, end } => ObstacleShape::Line {
            start: start + d,
            end: end + d,
        },
    }
}

#[inline]
fn add_scalar_residual(d: &mut StageDerivs, r: f64, gx: Vector4<f64>, gu: Vector2<f64>) {
    d.cost += 0.5 * r * r;
    d.lx += gx * r;
    d.lu += gu * r;
    d.lxx += gx * RowVector4::new(gx[0], gx[1], gx[2], gx[3]);
    d.luu += gu * RowVector2::new(gu[0], gu[1]);
    d.lux += gu * RowVector4::new(gx[0], gx[1], gx[2], gx[3]);
}

/// Compute all stage-cost derivatives at stage `k` (time `t_stage = k·dt`).
pub fn stage_derivs(
    k: usize,
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
    world: &WorldSnapshot,
    dt: f64,
) -> StageDerivs {
    let mut d = StageDerivs::zeros();
    let pos = Vec2::new(x[0], x[1]);
    let vel = Vec2::new(x[2], x[3]);
    let t_stage = k as f64 * dt;

    // -- Position tracking --
    let (p_ref, v_ref_opt) = eval_reference(&target.reference, t_stage);
    let w_p = target.weights.position.max(0.0);
    if w_p > 0.0 {
        let sqrt_w = w_p.sqrt();
        // residual x-component
        {
            let r = sqrt_w * (pos.x - p_ref.x);
            let gx = sqrt_w * Vector4::new(1.0, 0.0, 0.0, 0.0);
            let gu = Vector2::zeros();
            add_scalar_residual(&mut d, r, gx, gu);
        }
        // residual y-component
        {
            let r = sqrt_w * (pos.y - p_ref.y);
            let gx = sqrt_w * Vector4::new(0.0, 1.0, 0.0, 0.0);
            let gu = Vector2::zeros();
            add_scalar_residual(&mut d, r, gx, gu);
        }
    }

    // -- Velocity tracking (only if reference supplies velocity) --
    if let Some(v_ref) = v_ref_opt {
        let w_v = target.weights.velocity.max(0.0);
        if w_v > 0.0 {
            let sqrt_w = w_v.sqrt();
            {
                let r = sqrt_w * (vel.x - v_ref.x);
                let gx = sqrt_w * Vector4::new(0.0, 0.0, 1.0, 0.0);
                let gu = Vector2::zeros();
                add_scalar_residual(&mut d, r, gx, gu);
            }
            {
                let r = sqrt_w * (vel.y - v_ref.y);
                let gx = sqrt_w * Vector4::new(0.0, 0.0, 0.0, 1.0);
                let gu = Vector2::zeros();
                add_scalar_residual(&mut d, r, gx, gu);
            }
        }
    }

    // -- Control smoothness (u - u_prev), scaled by (1 - aggressiveness) --
    let smooth_scale = (1.0 - target.aggressiveness).clamp(0.0, 1.0);
    let w_sm = target.weights.control_smoothness.max(0.0) * smooth_scale;
    if w_sm > 0.0 {
        let sqrt_w = w_sm.sqrt();
        {
            let r = sqrt_w * (u[0] - u_prev[0]);
            let gx = Vector4::zeros();
            let gu = sqrt_w * Vector2::new(1.0, 0.0);
            add_scalar_residual(&mut d, r, gx, gu);
        }
        {
            let r = sqrt_w * (u[1] - u_prev[1]);
            let gx = Vector4::zeros();
            let gu = sqrt_w * Vector2::new(0.0, 1.0);
            add_scalar_residual(&mut d, r, gx, gu);
        }
    }

    // -- Obstacle barriers (predicted forward with constant velocity) --
    let care = target.care.max(0.0);
    let w_obs_base = target.weights.obstacle.max(0.0) * care;
    if w_obs_base > 0.0 {
        for obs in &world.obstacles {
            let w = w_obs_base * obs.weight_scale.max(0.0);
            if w == 0.0 {
                continue;
            }
            let sqrt_w = w.sqrt();
            let predicted = predict_shape(&obs.shape, obs.velocity, t_stage);
            let (b, grad_pos) = obstacle_residual(pos, &predicted, obs.safe_dist, obs.no_cost_dist);
            if b == 0.0 {
                continue;
            }
            let r = sqrt_w * b;
            let gx = sqrt_w * Vector4::new(grad_pos.x, grad_pos.y, 0.0, 0.0);
            let gu = Vector2::zeros();
            add_scalar_residual(&mut d, r, gx, gu);
        }
    }

    // -- Field boundary soft walls (4 half-planes) --
    let w_fb = target.weights.field_boundary.max(0.0) * care;
    if w_fb > 0.0 {
        let sqrt_w = w_fb.sqrt();
        let fb = &world.field_bounds;
        let margin = fb.margin.max(1.0);
        let walls: [(f64, Vec2); 4] = [
            (pos.x + fb.half_length, Vec2::new(1.0, 0.0)),  // left
            (fb.half_length - pos.x, Vec2::new(-1.0, 0.0)), // right
            (pos.y + fb.half_width, Vec2::new(0.0, 1.0)),   // bottom
            (fb.half_width - pos.y, Vec2::new(0.0, -1.0)),  // top
        ];
        for (d_plane, grad_plane) in walls {
            let (b, grad_pos) = halfplane_residual(d_plane, grad_plane, 0.0, margin);
            if b == 0.0 {
                continue;
            }
            let r = sqrt_w * b;
            let gx = sqrt_w * Vector4::new(grad_pos.x, grad_pos.y, 0.0, 0.0);
            let gu = Vector2::zeros();
            add_scalar_residual(&mut d, r, gx, gu);
        }
    }

    d
}

/// Terminal cost derivatives. Dispatches on `TerminalMode`.
pub fn terminal_derivs(x_n: &Vector4<f64>, target: &MpcTarget) -> TerminalDerivs {
    let mut d = TerminalDerivs::zeros();
    let w = target.weights.terminal.max(0.0);
    if w == 0.0 {
        return d;
    }
    let sqrt_w = w.sqrt();
    let pos = Vec2::new(x_n[0], x_n[1]);
    let vel = Vec2::new(x_n[2], x_n[3]);

    let add = |d: &mut TerminalDerivs, r: f64, gx: Vector4<f64>| {
        d.cost += 0.5 * r * r;
        d.lx += gx * r;
        d.lxx += gx * RowVector4::new(gx[0], gx[1], gx[2], gx[3]);
    };

    let (target_p, target_v_opt) = match &target.terminal {
        TerminalMode::Position { p } => (*p, None),
        TerminalMode::PositionAndVelocity { p, v } => (*p, Some(*v)),
        TerminalMode::RelativeVelocity { target_p, target_v } => (*target_p, Some(*target_v)),
    };

    // Position residual
    {
        let r = sqrt_w * (pos.x - target_p.x);
        let gx = sqrt_w * Vector4::new(1.0, 0.0, 0.0, 0.0);
        add(&mut d, r, gx);
    }
    {
        let r = sqrt_w * (pos.y - target_p.y);
        let gx = sqrt_w * Vector4::new(0.0, 1.0, 0.0, 0.0);
        add(&mut d, r, gx);
    }
    if let Some(target_v) = target_v_opt {
        {
            let r = sqrt_w * (vel.x - target_v.x);
            let gx = sqrt_w * Vector4::new(0.0, 0.0, 1.0, 0.0);
            add(&mut d, r, gx);
        }
        {
            let r = sqrt_w * (vel.y - target_v.y);
            let gx = sqrt_w * Vector4::new(0.0, 0.0, 0.0, 1.0);
            add(&mut d, r, gx);
        }
    }
    d
}

/// Pure scalar stage cost (no derivatives) — handy for line-search rollouts
/// and for finite-difference gradient tests.
pub fn stage_cost_scalar(
    k: usize,
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
    world: &WorldSnapshot,
    dt: f64,
) -> f64 {
    let mut c = 0.0;
    let pos = Vec2::new(x[0], x[1]);
    let vel = Vec2::new(x[2], x[3]);
    let t_stage = k as f64 * dt;

    let (p_ref, v_ref_opt) = eval_reference(&target.reference, t_stage);
    let w_p = target.weights.position.max(0.0);
    if w_p > 0.0 {
        c += 0.5 * w_p * ((pos.x - p_ref.x).powi(2) + (pos.y - p_ref.y).powi(2));
    }
    if let Some(v_ref) = v_ref_opt {
        let w_v = target.weights.velocity.max(0.0);
        if w_v > 0.0 {
            c += 0.5 * w_v * ((vel.x - v_ref.x).powi(2) + (vel.y - v_ref.y).powi(2));
        }
    }
    let smooth_scale = (1.0 - target.aggressiveness).clamp(0.0, 1.0);
    let w_sm = target.weights.control_smoothness.max(0.0) * smooth_scale;
    if w_sm > 0.0 {
        c += 0.5 * w_sm * ((u[0] - u_prev[0]).powi(2) + (u[1] - u_prev[1]).powi(2));
    }
    let care = target.care.max(0.0);
    let w_obs_base = target.weights.obstacle.max(0.0) * care;
    if w_obs_base > 0.0 {
        for obs in &world.obstacles {
            let w = w_obs_base * obs.weight_scale.max(0.0);
            if w == 0.0 {
                continue;
            }
            let predicted = predict_shape(&obs.shape, obs.velocity, t_stage);
            let (b, _) = obstacle_residual(pos, &predicted, obs.safe_dist, obs.no_cost_dist);
            c += 0.5 * w * b * b;
        }
    }
    let w_fb = target.weights.field_boundary.max(0.0) * care;
    if w_fb > 0.0 {
        let fb = &world.field_bounds;
        let margin = fb.margin.max(1.0);
        let walls = [
            pos.x + fb.half_length,
            fb.half_length - pos.x,
            pos.y + fb.half_width,
            fb.half_width - pos.y,
        ];
        for d_plane in walls {
            let (b, _) = crate::barrier::barrier_scalar(d_plane, 0.0, margin);
            c += 0.5 * w_fb * b * b;
        }
    }
    c
}

pub fn terminal_cost_scalar(x_n: &Vector4<f64>, target: &MpcTarget) -> f64 {
    let w = target.weights.terminal.max(0.0);
    if w == 0.0 {
        return 0.0;
    }
    let pos = Vec2::new(x_n[0], x_n[1]);
    let vel = Vec2::new(x_n[2], x_n[3]);
    let (target_p, target_v_opt) = match &target.terminal {
        TerminalMode::Position { p } => (*p, None),
        TerminalMode::PositionAndVelocity { p, v } => (*p, Some(*v)),
        TerminalMode::RelativeVelocity { target_p, target_v } => (*target_p, Some(*target_v)),
    };
    let mut c = 0.5 * w * ((pos.x - target_p.x).powi(2) + (pos.y - target_p.y).powi(2));
    if let Some(tv) = target_v_opt {
        c += 0.5 * w * ((vel.x - tv.x).powi(2) + (vel.y - tv.y).powi(2));
    }
    c
}

// suppress the unused-import warning when the helpers aren't referenced in release paths.
#[allow(dead_code)]
fn _unused_marker(_: &PredictedObstacle) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CostWeights, FieldBounds, MpcTarget, PredictedObstacle, ReferenceTrajectory, TerminalMode,
    };
    use approx::assert_abs_diff_eq;

    fn world_no_obstacles() -> WorldSnapshot {
        WorldSnapshot {
            obstacles: vec![],
            field_bounds: FieldBounds::centered(9000.0, 6000.0, 1000.0, 2000.0),
        }
    }

    fn target_goto(p: Vec2) -> MpcTarget {
        MpcTarget {
            reference: ReferenceTrajectory::StaticPoint(p),
            terminal: TerminalMode::Position { p },
            weights: CostWeights::default(),
            care: 1.0,
            aggressiveness: 0.0,
        }
    }

    #[test]
    fn terminal_position_zero_at_target() {
        let p = Vec2::new(1000.0, 500.0);
        let t = target_goto(p);
        let x = Vector4::new(p.x, p.y, 0.0, 0.0);
        let d = terminal_derivs(&x, &t);
        assert_abs_diff_eq!(d.cost, 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn terminal_relative_velocity_zero_at_match() {
        let p = Vec2::new(500.0, -300.0);
        let v = Vec2::new(1200.0, 800.0);
        let mut t = target_goto(p);
        t.terminal = TerminalMode::RelativeVelocity {
            target_p: p,
            target_v: v,
        };
        let x = Vector4::new(p.x, p.y, v.x, v.y);
        let d = terminal_derivs(&x, &t);
        assert_abs_diff_eq!(d.cost, 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn obstacle_barrier_zero_outside_no_cost_zone() {
        let target = target_goto(Vec2::new(0.0, 0.0));
        let world = WorldSnapshot {
            obstacles: vec![PredictedObstacle {
                shape: ObstacleShape::Circle {
                    center: Vec2::new(0.0, 0.0),
                    radius: 100.0,
                },
                velocity: Vec2::zeros(),
                safe_dist: 200.0,
                no_cost_dist: 400.0,
                weight_scale: 1.0,
            }],
            field_bounds: FieldBounds::centered(90_000.0, 60_000.0, 1000.0, 2000.0),
        };
        // Robot far away from the obstacle.
        let x = Vector4::new(10_000.0, 0.0, 0.0, 0.0);
        let u = Vector2::zeros();
        let u_prev = Vector2::zeros();
        let c = stage_cost_scalar(0, &x, &u, &u_prev, &target, &world, 0.06);
        // Cost should come only from position tracking, not obstacle barrier.
        let expected = 0.5 * target.weights.position * (10_000.0_f64).powi(2);
        assert_abs_diff_eq!(c, expected, epsilon = 1.0e-6);
    }

    fn finite_diff_gradient(
        k: usize,
        x: &Vector4<f64>,
        u: &Vector2<f64>,
        u_prev: &Vector2<f64>,
        target: &MpcTarget,
        world: &WorldSnapshot,
        dt: f64,
    ) -> (Vector4<f64>, Vector2<f64>) {
        let eps = 1.0e-3;
        let mut gx = Vector4::zeros();
        let mut gu = Vector2::zeros();
        for i in 0..4 {
            let mut xp = *x;
            let mut xm = *x;
            xp[i] += eps;
            xm[i] -= eps;
            let cp = stage_cost_scalar(k, &xp, u, u_prev, target, world, dt);
            let cm = stage_cost_scalar(k, &xm, u, u_prev, target, world, dt);
            gx[i] = (cp - cm) / (2.0 * eps);
        }
        for i in 0..2 {
            let mut up = *u;
            let mut um = *u;
            up[i] += eps;
            um[i] -= eps;
            let cp = stage_cost_scalar(k, x, &up, u_prev, target, world, dt);
            let cm = stage_cost_scalar(k, x, &um, u_prev, target, world, dt);
            gu[i] = (cp - cm) / (2.0 * eps);
        }
        (gx, gu)
    }

    #[test]
    fn stage_gradient_matches_finite_diff_no_obstacles() {
        let target = {
            let mut t = target_goto(Vec2::new(2000.0, 1000.0));
            t.aggressiveness = 0.2;
            t
        };
        let world = world_no_obstacles();
        let x = Vector4::new(500.0, -200.0, 1000.0, -500.0);
        let u = Vector2::new(1500.0, 800.0);
        let u_prev = Vector2::new(1200.0, 600.0);
        let dt = 0.06;
        let d = stage_derivs(3, &x, &u, &u_prev, &target, &world, dt);
        let (fd_gx, fd_gu) = finite_diff_gradient(3, &x, &u, &u_prev, &target, &world, dt);
        for i in 0..4 {
            assert_abs_diff_eq!(d.lx[i], fd_gx[i], epsilon = 1.0e-3);
        }
        for i in 0..2 {
            assert_abs_diff_eq!(d.lu[i], fd_gu[i], epsilon = 1.0e-3);
        }
    }

    #[test]
    fn stage_gradient_matches_finite_diff_with_obstacle_in_ramp() {
        let target = target_goto(Vec2::new(1500.0, 0.0));
        let world = WorldSnapshot {
            obstacles: vec![PredictedObstacle {
                shape: ObstacleShape::Circle {
                    center: Vec2::new(750.0, 100.0),
                    radius: 90.0,
                },
                velocity: Vec2::zeros(),
                safe_dist: 180.0,
                no_cost_dist: 360.0,
                weight_scale: 1.0,
            }],
            field_bounds: FieldBounds::centered(9000.0, 6000.0, 1000.0, 2000.0),
        };
        let x = Vector4::new(700.0, 250.0, 500.0, -100.0);
        let u = Vector2::new(1200.0, 0.0);
        let u_prev = Vector2::new(1000.0, 100.0);
        let dt = 0.06;
        let d = stage_derivs(0, &x, &u, &u_prev, &target, &world, dt);
        let (fd_gx, fd_gu) = finite_diff_gradient(0, &x, &u, &u_prev, &target, &world, dt);
        for i in 0..4 {
            assert_abs_diff_eq!(d.lx[i], fd_gx[i], epsilon = 2.0e-3);
        }
        for i in 0..2 {
            assert_abs_diff_eq!(d.lu[i], fd_gu[i], epsilon = 2.0e-3);
        }
    }

    #[test]
    fn terminal_gradient_matches_finite_diff_relvel() {
        let mut target = target_goto(Vec2::new(0.0, 0.0));
        target.terminal = TerminalMode::RelativeVelocity {
            target_p: Vec2::new(1500.0, -800.0),
            target_v: Vec2::new(500.0, 200.0),
        };
        let x = Vector4::new(1000.0, -200.0, 800.0, 100.0);
        let d = terminal_derivs(&x, &target);
        let eps = 1.0e-3;
        for i in 0..4 {
            let mut xp = x;
            let mut xm = x;
            xp[i] += eps;
            xm[i] -= eps;
            let cp = terminal_cost_scalar(&xp, &target);
            let cm = terminal_cost_scalar(&xm, &target);
            let fd = (cp - cm) / (2.0 * eps);
            assert_abs_diff_eq!(d.lx[i], fd, epsilon = 1.0e-4);
        }
    }

    #[test]
    fn reference_interpolation_linear() {
        let samples = vec![
            crate::types::TimedRef {
                t: 0.0,
                pos: Vec2::new(0.0, 0.0),
                vel: Some(Vec2::new(100.0, 0.0)),
            },
            crate::types::TimedRef {
                t: 1.0,
                pos: Vec2::new(1000.0, 500.0),
                vel: Some(Vec2::new(200.0, 100.0)),
            },
        ];
        let r = ReferenceTrajectory::Timed(samples);
        let (p, v) = eval_reference(&r, 0.5);
        assert_abs_diff_eq!(p.x, 500.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(p.y, 250.0, epsilon = 1.0e-9);
        let v = v.unwrap();
        assert_abs_diff_eq!(v.x, 150.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(v.y, 50.0, epsilon = 1.0e-9);
    }
}
