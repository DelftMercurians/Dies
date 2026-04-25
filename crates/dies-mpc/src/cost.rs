//! Quadratic stage + terminal cost for the iLQR solver.
//!
//! Stage cost (per stage `k`):
//! ```text
//! L_k = ½ · w_p · ‖pos − target_p‖²
//!     + ½ · w_v · ‖vel − target_v‖²
//!     + ½ · w_u · ‖u‖²
//!     + ½ · w_du · ‖u − u_prev‖²
//! ```
//! Terminal cost (at stage `N`):
//! ```text
//! L_N = ½ · w_term_p · ‖pos_N − target_p‖²
//!     + ½ · w_term_v · ‖vel_N − target_v‖²
//! ```
//!
//! Every term is quadratic; the Hessian is exact. The `w_u‖u‖²` term is the
//! one that makes `Q_uu` non-degenerate at the target — without it, iLQR
//! feedback gains explode (1 mm of state error → 500 mm/s of command).

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

use crate::types::MpcTarget;

#[derive(Clone, Debug)]
pub struct StageDerivs {
    pub cost: f64,
    pub lx: Vector4<f64>,
    pub lu: Vector2<f64>,
    pub lxx: Matrix4<f64>,
    pub luu: Matrix2<f64>,
    /// `∂²L / (∂u ∂x)`. Always zero for this cost (no cross terms), but kept
    /// in the API so the solver stays generic.
    pub lux: Matrix2x4<f64>,
}

#[derive(Clone, Debug)]
pub struct TerminalDerivs {
    pub cost: f64,
    pub lx: Vector4<f64>,
    pub lxx: Matrix4<f64>,
}

/// Compute stage-cost value + derivatives at `(x, u, u_prev)`.
pub fn stage_derivs(
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
) -> StageDerivs {
    let mut cost = 0.0;
    let mut lx = Vector4::zeros();
    let mut lu = Vector2::zeros();
    let mut lxx = Matrix4::zeros();
    let mut luu = Matrix2::zeros();

    let w = &target.weights;

    // pos residual
    if w.position > 0.0 {
        let dx = x[0] - target.p.x;
        let dy = x[1] - target.p.y;
        cost += 0.5 * w.position * (dx * dx + dy * dy);
        lx[0] += w.position * dx;
        lx[1] += w.position * dy;
        lxx[(0, 0)] += w.position;
        lxx[(1, 1)] += w.position;
    }

    // vel residual
    if w.velocity > 0.0 {
        let dvx = x[2] - target.v.x;
        let dvy = x[3] - target.v.y;
        cost += 0.5 * w.velocity * (dvx * dvx + dvy * dvy);
        lx[2] += w.velocity * dvx;
        lx[3] += w.velocity * dvy;
        lxx[(2, 2)] += w.velocity;
        lxx[(3, 3)] += w.velocity;
    }

    // ‖u‖²  — the term that bounds Q_uu.
    if w.control > 0.0 {
        cost += 0.5 * w.control * (u[0] * u[0] + u[1] * u[1]);
        lu[0] += w.control * u[0];
        lu[1] += w.control * u[1];
        luu[(0, 0)] += w.control;
        luu[(1, 1)] += w.control;
    }

    // ‖u − u_prev‖²
    if w.control_smoothness > 0.0 {
        let dux = u[0] - u_prev[0];
        let duy = u[1] - u_prev[1];
        cost += 0.5 * w.control_smoothness * (dux * dux + duy * duy);
        lu[0] += w.control_smoothness * dux;
        lu[1] += w.control_smoothness * duy;
        luu[(0, 0)] += w.control_smoothness;
        luu[(1, 1)] += w.control_smoothness;
    }

    StageDerivs {
        cost,
        lx,
        lu,
        lxx,
        luu,
        lux: Matrix2x4::zeros(),
    }
}

/// Pure scalar stage cost — used by line-search rollouts.
pub fn stage_cost_scalar(
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
) -> f64 {
    let w = &target.weights;
    let mut c = 0.0;
    if w.position > 0.0 {
        let dx = x[0] - target.p.x;
        let dy = x[1] - target.p.y;
        c += 0.5 * w.position * (dx * dx + dy * dy);
    }
    if w.velocity > 0.0 {
        let dvx = x[2] - target.v.x;
        let dvy = x[3] - target.v.y;
        c += 0.5 * w.velocity * (dvx * dvx + dvy * dvy);
    }
    if w.control > 0.0 {
        c += 0.5 * w.control * (u[0] * u[0] + u[1] * u[1]);
    }
    if w.control_smoothness > 0.0 {
        let dux = u[0] - u_prev[0];
        let duy = u[1] - u_prev[1];
        c += 0.5 * w.control_smoothness * (dux * dux + duy * duy);
    }
    c
}

/// Terminal cost value + derivatives at `x_N`.
pub fn terminal_derivs(x_n: &Vector4<f64>, target: &MpcTarget) -> TerminalDerivs {
    let w = &target.weights;
    let mut cost = 0.0;
    let mut lx = Vector4::zeros();
    let mut lxx = Matrix4::zeros();

    if w.terminal_position > 0.0 {
        let dx = x_n[0] - target.p.x;
        let dy = x_n[1] - target.p.y;
        cost += 0.5 * w.terminal_position * (dx * dx + dy * dy);
        lx[0] += w.terminal_position * dx;
        lx[1] += w.terminal_position * dy;
        lxx[(0, 0)] += w.terminal_position;
        lxx[(1, 1)] += w.terminal_position;
    }

    if w.terminal_velocity > 0.0 {
        let dvx = x_n[2] - target.v.x;
        let dvy = x_n[3] - target.v.y;
        cost += 0.5 * w.terminal_velocity * (dvx * dvx + dvy * dvy);
        lx[2] += w.terminal_velocity * dvx;
        lx[3] += w.terminal_velocity * dvy;
        lxx[(2, 2)] += w.terminal_velocity;
        lxx[(3, 3)] += w.terminal_velocity;
    }

    TerminalDerivs { cost, lx, lxx }
}

pub fn terminal_cost_scalar(x_n: &Vector4<f64>, target: &MpcTarget) -> f64 {
    let w = &target.weights;
    let mut c = 0.0;
    if w.terminal_position > 0.0 {
        let dx = x_n[0] - target.p.x;
        let dy = x_n[1] - target.p.y;
        c += 0.5 * w.terminal_position * (dx * dx + dy * dy);
    }
    if w.terminal_velocity > 0.0 {
        let dvx = x_n[2] - target.v.x;
        let dvy = x_n[3] - target.v.y;
        c += 0.5 * w.terminal_velocity * (dvx * dvx + dvy * dvy);
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec2;
    use approx::assert_abs_diff_eq;

    fn target() -> MpcTarget {
        MpcTarget::goto(Vec2::new(1000.0, 500.0))
    }

    #[test]
    fn zero_at_target_with_zero_control() {
        let t = target();
        let x = Vector4::new(t.p.x, t.p.y, t.v.x, t.v.y);
        let u = Vector2::zeros();
        assert_abs_diff_eq!(
            stage_cost_scalar(&x, &u, &u, &t),
            0.0,
            epsilon = 1.0e-12
        );
        assert_abs_diff_eq!(terminal_cost_scalar(&x, &t), 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn stage_derivs_match_finite_diff() {
        let t = target();
        let x = Vector4::new(800.0, 200.0, -300.0, 150.0);
        let u = Vector2::new(450.0, -200.0);
        let u_prev = Vector2::new(100.0, 50.0);
        let d = stage_derivs(&x, &u, &u_prev, &t);
        let eps = 1.0e-4;
        for j in 0..4 {
            let mut xp = x;
            let mut xm = x;
            xp[j] += eps;
            xm[j] -= eps;
            let fd = (stage_cost_scalar(&xp, &u, &u_prev, &t)
                - stage_cost_scalar(&xm, &u, &u_prev, &t))
                / (2.0 * eps);
            assert_abs_diff_eq!(d.lx[j], fd, epsilon = 1.0e-5);
        }
        for j in 0..2 {
            let mut up = u;
            let mut um = u;
            up[j] += eps;
            um[j] -= eps;
            let fd = (stage_cost_scalar(&x, &up, &u_prev, &t)
                - stage_cost_scalar(&x, &um, &u_prev, &t))
                / (2.0 * eps);
            assert_abs_diff_eq!(d.lu[j], fd, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn terminal_derivs_match_finite_diff() {
        let t = target();
        let x = Vector4::new(800.0, 200.0, -300.0, 150.0);
        let d = terminal_derivs(&x, &t);
        let eps = 1.0e-4;
        for j in 0..4 {
            let mut xp = x;
            let mut xm = x;
            xp[j] += eps;
            xm[j] -= eps;
            let fd = (terminal_cost_scalar(&xp, &t) - terminal_cost_scalar(&xm, &t)) / (2.0 * eps);
            assert_abs_diff_eq!(d.lx[j], fd, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn control_term_creates_q_uu_floor() {
        // The whole point of the `w.control` term: ensure `luu` has at least
        // `w.control` on its diagonal even at (x, u, u_prev) = 0.
        let t = target();
        let x = Vector4::zeros();
        let u = Vector2::zeros();
        let d = stage_derivs(&x, &u, &u, &t);
        let expected = t.weights.control + t.weights.control_smoothness;
        assert_abs_diff_eq!(d.luu[(0, 0)], expected, epsilon = 1.0e-12);
        assert_abs_diff_eq!(d.luu[(1, 1)], expected, epsilon = 1.0e-12);
    }
}
