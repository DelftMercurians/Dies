//! Stage cost for the iLQR solver: translational tracking + heading tracking.
//!
//! Stage cost (per stage `k`):
//! ```text
//! L_k = ½ · w_p · ‖pos − target_p‖²
//!     + ½ · w_v · ‖vel − target_v‖²
//!     + ½ · w_u · ‖u_trans‖²
//!     + ½ · w_du · ‖u_trans − u_prev_trans‖²
//!     +     w_yaw · (1 − cos(θ − θ_d))
//!     + ½ · w_yawctrl · (θ_cmd − θ)²
//! ```
//!
//! Translational terms are quadratic (exact Hessian); the heading terms add
//! curvature on the `θ` / `θ_cmd` axes. The `w_u‖u_trans‖²` and `w_yawctrl`
//! terms keep `Q_uu` non-degenerate so iLQR feedback gains stay bounded.
//! Obstacle barriers (position-only) are added on top.

use nalgebra::{Matrix3, Matrix3x5, Matrix5, Vector3, Vector5};

use crate::obstacle;
use crate::types::{MpcTarget, Vec2};

#[derive(Clone, Debug)]
pub struct StageDerivs {
    pub cost: f64,
    pub lx: Vector5<f64>,
    pub lu: Vector3<f64>,
    pub lxx: Matrix5<f64>,
    pub luu: Matrix3<f64>,
    /// `∂²L / (∂u ∂x)`. Couples `θ_cmd` and `θ` through the turn term.
    pub lux: Matrix3x5<f64>,
}

/// Compute stage-cost value + derivatives at `(x, u, u_prev)`. `t` is the stage
/// time [s] from the start of the horizon, used to drift moving obstacles.
pub fn stage_derivs(
    x: &Vector5<f64>,
    u: &Vector3<f64>,
    u_prev: &Vector3<f64>,
    target: &MpcTarget,
    t: f64,
) -> StageDerivs {
    let (mut cost, mut lx, lu, mut lxx, luu, lux) =
        crate::generated::cost::stage_derivs(x, u, u_prev, target);

    // Obstacle barriers depend only on position, so they add into the position
    // block of the gradient (lx[0..2]) and Hessian (lxx[0..2, 0..2]) only.
    if !target.obstacles.is_empty() {
        let p = Vec2::new(x[0], x[1]);
        let (oc, og, oh) = obstacle::penalty_derivs(&target.obstacles, p, t);
        cost += oc;
        lx[0] += og[0];
        lx[1] += og[1];
        lxx[(0, 0)] += oh[(0, 0)];
        lxx[(0, 1)] += oh[(0, 1)];
        lxx[(1, 0)] += oh[(1, 0)];
        lxx[(1, 1)] += oh[(1, 1)];
    }

    StageDerivs {
        cost,
        lx,
        lu,
        lxx,
        luu,
        lux,
    }
}

/// Pure scalar stage cost — used by line-search rollouts.
pub fn stage_cost_scalar(
    x: &Vector5<f64>,
    u: &Vector3<f64>,
    u_prev: &Vector3<f64>,
    target: &MpcTarget,
    t: f64,
) -> f64 {
    let mut cost = crate::generated::cost::stage_cost_scalar(x, u, u_prev, target);
    if !target.obstacles.is_empty() {
        cost += obstacle::penalty_scalar(&target.obstacles, Vec2::new(x[0], x[1]), t);
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Vec2;
    use approx::assert_abs_diff_eq;

    fn target() -> MpcTarget {
        let mut t = MpcTarget::goto(Vec2::new(1000.0, 500.0));
        t.heading = 0.4;
        t
    }

    #[test]
    fn zero_at_target_with_zero_heading_error() {
        // At the target position+heading with zero control error every residual
        // vanishes. θ = θ_d and θ_cmd = θ ⇒ heading terms are zero too.
        let mut t = target();
        t.heading = 0.0;
        let x = Vector5::new(t.p.x, t.p.y, t.v.x, t.v.y, 0.0);
        let u = Vector3::zeros();
        assert_abs_diff_eq!(
            stage_cost_scalar(&x, &u, &u, &t, 0.0),
            0.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn stage_derivs_match_finite_diff() {
        let t = target();
        let x = Vector5::new(800.0, 200.0, -300.0, 150.0, 0.2);
        let u = Vector3::new(450.0, -200.0, 0.6);
        let u_prev = Vector3::new(100.0, 50.0, 0.1);
        let d = stage_derivs(&x, &u, &u_prev, &t, 0.0);
        let eps = 1.0e-4;
        for j in 0..5 {
            let mut xp = x;
            let mut xm = x;
            xp[j] += eps;
            xm[j] -= eps;
            let fd = (stage_cost_scalar(&xp, &u, &u_prev, &t, 0.0)
                - stage_cost_scalar(&xm, &u, &u_prev, &t, 0.0))
                / (2.0 * eps);
            assert_abs_diff_eq!(d.lx[j], fd, epsilon = 1.0e-5);
        }
        for j in 0..3 {
            let mut up = u;
            let mut um = u;
            up[j] += eps;
            um[j] -= eps;
            let fd = (stage_cost_scalar(&x, &up, &u_prev, &t, 0.0)
                - stage_cost_scalar(&x, &um, &u_prev, &t, 0.0))
                / (2.0 * eps);
            assert_abs_diff_eq!(d.lu[j], fd, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn stage_derivs_with_obstacle_match_finite_diff() {
        // Gradient of the combined tracking + obstacle cost must match finite
        // differences (the Gauss-Newton Hessian is intentionally not exact, so
        // we only check the gradient — same as the tracking-only test).
        use crate::obstacle::{Obstacle, ObstacleShape};
        let mut t = target();
        t.obstacles = vec![Obstacle::fixed(
            ObstacleShape::Circle {
                center: Vec2::new(700.0, 250.0),
                radius: 180.0,
            },
            5.0e-2,
            250.0,
        )];
        // Place the robot inside the influence shell so the barrier is active.
        let x = Vector5::new(820.0, 220.0, -300.0, 150.0, 0.2);
        let u = Vector3::new(450.0, -200.0, 0.6);
        let u_prev = Vector3::new(100.0, 50.0, 0.1);
        let d = stage_derivs(&x, &u, &u_prev, &t, 0.0);
        let eps = 1.0e-3;
        for j in 0..5 {
            let mut xp = x;
            let mut xm = x;
            xp[j] += eps;
            xm[j] -= eps;
            let fd = (stage_cost_scalar(&xp, &u, &u_prev, &t, 0.0)
                - stage_cost_scalar(&xm, &u, &u_prev, &t, 0.0))
                / (2.0 * eps);
            assert_abs_diff_eq!(d.lx[j], fd, epsilon = 1.0e-4);
        }
    }

    #[test]
    fn control_terms_create_q_uu_floor() {
        // Translational control terms floor `luu` on the velocity axes; the
        // `heading_control` term floors it on the `θ_cmd` axis. Without these
        // floors iLQR feedback gains would blow up.
        let t = target();
        let x = Vector5::zeros();
        let u = Vector3::zeros();
        let d = stage_derivs(&x, &u, &u, &t, 0.0);
        let trans = t.weights.control + t.weights.control_smoothness;
        assert_abs_diff_eq!(d.luu[(0, 0)], trans, epsilon = 1.0e-12);
        assert_abs_diff_eq!(d.luu[(1, 1)], trans, epsilon = 1.0e-12);
        assert_abs_diff_eq!(d.luu[(2, 2)], t.weights.heading_control, epsilon = 1.0e-12);
    }
}
