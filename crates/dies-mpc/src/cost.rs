//! Quadratic stage cost for the iLQR solver.
//!
//! Stage cost (per stage `k`):
//! ```text
//! L_k = ½ · w_p · ‖pos − target_p‖²
//!     + ½ · w_v · ‖vel − target_v‖²
//!     + ½ · w_u · ‖u‖²
//!     + ½ · w_du · ‖u − u_prev‖²
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

/// Compute stage-cost value + derivatives at `(x, u, u_prev)`.
pub fn stage_derivs(
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
) -> StageDerivs {
    let (cost, lx, lu, lxx, luu, lux) = crate::generated::cost::stage_derivs(x, u, u_prev, target);
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
    x: &Vector4<f64>,
    u: &Vector2<f64>,
    u_prev: &Vector2<f64>,
    target: &MpcTarget,
) -> f64 {
    crate::generated::cost::stage_cost_scalar(x, u, u_prev, target)
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
        assert_abs_diff_eq!(stage_cost_scalar(&x, &u, &u, &t), 0.0, epsilon = 1.0e-12);
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
