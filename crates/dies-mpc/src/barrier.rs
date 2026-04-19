//! Soft barrier function shared by obstacle, field-boundary and goal-area costs.
//!
//! The barrier returns a *residual* `b(d)` such that the corresponding stage
//! cost contribution is `½ · weight · b²`. This matches the Gauss-Newton
//! formulation used by the cost module: `grad = weight · b · ∇b`,
//! `hess ≈ weight · ∇b · ∇bᵀ`.
//!
//! Shape:
//! ```text
//! b(d) = max(0, (no_cost − d) / (no_cost − safe))
//! ```
//! — linear in `d`, zero above `no_cost`, with a single kink at `d = no_cost`
//! where the value and first derivative are both zero (so the penalty is
//! C¹ there). For `d < safe` the residual keeps growing linearly, which
//! produces the desired quadratic cost blow-up during penetration while
//! keeping all derivatives bounded and smooth enough for iLQR.

use crate::types::{ObstacleShape, Vec2};

/// Residual value and its derivative with respect to the signed distance `d`.
///
/// Returns `(b, db/dd)`. Zero above `no_cost`; linearly growing as `d`
/// decreases. The ramp slope is `-1 / (no_cost − safe)`.
#[inline]
pub fn barrier_scalar(d: f64, safe: f64, no_cost: f64) -> (f64, f64) {
    let range = (no_cost - safe).max(1.0e-6);
    let u = (no_cost - d) / range;
    if u <= 0.0 {
        (0.0, 0.0)
    } else {
        (u, -1.0 / range)
    }
}

/// Signed distance from a point to an obstacle shape. Outside: `d > 0` with
/// the gradient pointing away from the shape. Inside: `d < 0` with the
/// gradient still pointing outward (continuous extension, required so the
/// barrier gradient remains well-defined during penetration).
pub fn signed_distance(pos: Vec2, shape: &ObstacleShape) -> (f64, Vec2) {
    match shape {
        ObstacleShape::Circle { center, radius } => {
            let diff = pos - center;
            let dist = diff.norm();
            let eps = 1.0e-9;
            let dir = if dist > eps {
                diff / dist
            } else {
                Vec2::new(1.0, 0.0)
            };
            (dist - radius, dir)
        }
        ObstacleShape::Rectangle { min, max } => {
            let cx = 0.5 * (min.x + max.x);
            let cy = 0.5 * (min.y + max.y);
            let hx = 0.5 * (max.x - min.x);
            let hy = 0.5 * (max.y - min.y);
            let sx = if pos.x >= cx { 1.0 } else { -1.0 };
            let sy = if pos.y >= cy { 1.0 } else { -1.0 };
            let qx = (pos.x - cx).abs() - hx;
            let qy = (pos.y - cy).abs() - hy;
            let outside_x = qx.max(0.0);
            let outside_y = qy.max(0.0);
            let outside_norm = (outside_x * outside_x + outside_y * outside_y).sqrt();
            let inside = qx.max(qy).min(0.0);
            let d = outside_norm + inside;

            let grad = if outside_norm > 1.0e-9 {
                Vec2::new(
                    if qx > 0.0 {
                        sx * outside_x / outside_norm
                    } else {
                        0.0
                    },
                    if qy > 0.0 {
                        sy * outside_y / outside_norm
                    } else {
                        0.0
                    },
                )
            } else if qx > qy {
                Vec2::new(sx, 0.0)
            } else {
                Vec2::new(0.0, sy)
            };
            (d, grad)
        }
        ObstacleShape::Line { start, end } => {
            let seg = end - start;
            let seg_len_sq = seg.norm_squared();
            if seg_len_sq < 1.0e-12 {
                let diff = pos - start;
                let d = diff.norm();
                let dir = if d > 1.0e-9 {
                    diff / d
                } else {
                    Vec2::new(1.0, 0.0)
                };
                return (d, dir);
            }
            let t = ((pos - start).dot(&seg) / seg_len_sq).clamp(0.0, 1.0);
            let proj = start + seg * t;
            let diff = pos - proj;
            let d = diff.norm();
            let dir = if d > 1.0e-9 {
                diff / d
            } else {
                let perp = Vec2::new(-seg.y, seg.x);
                let n = perp.norm();
                if n > 1.0e-12 {
                    perp / n
                } else {
                    Vec2::new(1.0, 0.0)
                }
            };
            (d, dir)
        }
    }
}

/// Compute the obstacle-barrier residual and its gradient w.r.t. position.
///
/// Returns `(b, ∂b/∂pos)`. The gradient is zero in the no-cost region.
pub fn obstacle_residual(pos: Vec2, shape: &ObstacleShape, safe: f64, no_cost: f64) -> (f64, Vec2) {
    let (d, grad_d) = signed_distance(pos, shape);
    let (b, db_dd) = barrier_scalar(d, safe, no_cost);
    (b, grad_d * db_dd)
}

/// A single soft "stay inside" half-plane residual. `d_plane` is the signed
/// distance from the wall to the point, positive inside. `grad_plane` is the
/// (constant) gradient of `d_plane` w.r.t. position.
pub fn halfplane_residual(d_plane: f64, grad_plane: Vec2, safe: f64, no_cost: f64) -> (f64, Vec2) {
    let (b, db_dd) = barrier_scalar(d_plane, safe, no_cost);
    (b, grad_plane * db_dd)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn barrier_zero_above_no_cost() {
        let (b, db) = barrier_scalar(200.0, 100.0, 150.0);
        assert_eq!(b, 0.0);
        assert_eq!(db, 0.0);
    }

    #[test]
    fn barrier_linear_in_ramp() {
        let (b_far, _) = barrier_scalar(140.0, 100.0, 150.0);
        let (b_near, _) = barrier_scalar(110.0, 100.0, 150.0);
        assert_abs_diff_eq!(b_far, 0.2, epsilon = 1.0e-9);
        assert_abs_diff_eq!(b_near, 0.8, epsilon = 1.0e-9);
    }

    #[test]
    fn barrier_keeps_growing_below_safe() {
        let (b_safe, _) = barrier_scalar(100.0, 100.0, 150.0);
        let (b_penetrate, _) = barrier_scalar(50.0, 100.0, 150.0);
        assert_abs_diff_eq!(b_safe, 1.0, epsilon = 1.0e-9);
        assert!(b_penetrate > b_safe);
    }

    #[test]
    fn barrier_is_continuous_at_no_cost() {
        let (b_inside, _) = barrier_scalar(149.999, 100.0, 150.0);
        let (b_outside, _) = barrier_scalar(150.001, 100.0, 150.0);
        assert!(b_inside < 1.0e-3);
        assert_eq!(b_outside, 0.0);
    }

    #[test]
    fn circle_signed_distance_outside() {
        let shape = ObstacleShape::Circle {
            center: Vec2::new(0.0, 0.0),
            radius: 100.0,
        };
        let (d, grad) = signed_distance(Vec2::new(200.0, 0.0), &shape);
        assert_abs_diff_eq!(d, 100.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.x, 1.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.y, 0.0, epsilon = 1.0e-9);
    }

    #[test]
    fn circle_signed_distance_inside() {
        let shape = ObstacleShape::Circle {
            center: Vec2::new(0.0, 0.0),
            radius: 100.0,
        };
        let (d, grad) = signed_distance(Vec2::new(50.0, 0.0), &shape);
        assert_abs_diff_eq!(d, -50.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.x, 1.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.y, 0.0, epsilon = 1.0e-9);
    }

    #[test]
    fn rectangle_signed_distance_outside_corner() {
        let shape = ObstacleShape::Rectangle {
            min: Vec2::new(-100.0, -100.0),
            max: Vec2::new(100.0, 100.0),
        };
        let (d, _) = signed_distance(Vec2::new(200.0, 200.0), &shape);
        let expected = ((100.0_f64).powi(2) + (100.0_f64).powi(2)).sqrt();
        assert_abs_diff_eq!(d, expected, epsilon = 1.0e-9);
    }

    #[test]
    fn rectangle_signed_distance_inside() {
        let shape = ObstacleShape::Rectangle {
            min: Vec2::new(-100.0, -100.0),
            max: Vec2::new(100.0, 100.0),
        };
        let (d, _) = signed_distance(Vec2::new(0.0, 50.0), &shape);
        assert_abs_diff_eq!(d, -50.0, epsilon = 1.0e-9);
    }

    #[test]
    fn line_signed_distance_midpoint() {
        let shape = ObstacleShape::Line {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(100.0, 0.0),
        };
        let (d, grad) = signed_distance(Vec2::new(50.0, 30.0), &shape);
        assert_abs_diff_eq!(d, 30.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.x, 0.0, epsilon = 1.0e-9);
        assert_abs_diff_eq!(grad.y, 1.0, epsilon = 1.0e-9);
    }

    #[test]
    fn line_signed_distance_past_endpoint() {
        let shape = ObstacleShape::Line {
            start: Vec2::new(0.0, 0.0),
            end: Vec2::new(100.0, 0.0),
        };
        let (d, _) = signed_distance(Vec2::new(150.0, 0.0), &shape);
        assert_abs_diff_eq!(d, 50.0, epsilon = 1.0e-9);
    }

    #[test]
    fn circle_gradient_matches_finite_diff() {
        let shape = ObstacleShape::Circle {
            center: Vec2::new(10.0, 20.0),
            radius: 50.0,
        };
        let p = Vec2::new(80.0, 60.0);
        let (_, grad) = signed_distance(p, &shape);
        let eps = 1.0e-5;
        for axis in 0..2 {
            let mut pp = p;
            let mut pm = p;
            pp[axis] += eps;
            pm[axis] -= eps;
            let dp = signed_distance(pp, &shape).0;
            let dm = signed_distance(pm, &shape).0;
            let fd = (dp - dm) / (2.0 * eps);
            assert_abs_diff_eq!(grad[axis], fd, epsilon = 1.0e-4);
        }
    }

    #[test]
    fn rectangle_gradient_matches_finite_diff_outside() {
        let shape = ObstacleShape::Rectangle {
            min: Vec2::new(-100.0, -100.0),
            max: Vec2::new(100.0, 100.0),
        };
        let p = Vec2::new(150.0, 80.0);
        let (_, grad) = signed_distance(p, &shape);
        let eps = 1.0e-5;
        for axis in 0..2 {
            let mut pp = p;
            let mut pm = p;
            pp[axis] += eps;
            pm[axis] -= eps;
            let fd = (signed_distance(pp, &shape).0 - signed_distance(pm, &shape).0) / (2.0 * eps);
            assert_abs_diff_eq!(grad[axis], fd, epsilon = 1.0e-4);
        }
    }
}
