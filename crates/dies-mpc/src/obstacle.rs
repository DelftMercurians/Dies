//! Soft obstacle-avoidance penalties for the iLQR stage cost.
//!
//! Each obstacle contributes a one-sided quadratic-hinge barrier on the robot's
//! position: zero cost once the robot is at least `influence` mm clear of the
//! obstacle surface, growing as `½·w·(influence − clearance)²` as it penetrates
//! the influence shell. Penalties depend only on position, so they touch only
//! the position block of the stage gradient/Hessian.
//!
//! The Hessian returned is the Gauss-Newton approximation `w·∇d∇dᵀ`, which is
//! always positive semidefinite — keeping the iLQR backward pass well-behaved
//! even though the exact barrier Hessian is indefinite in the keep-out region.
//!
//! Obstacles are evaluated in the solver's (team-relative) frame and may carry a
//! constant-velocity drift so a moving robot is extrapolated forward along the
//! horizon (clamped to `vel_cap_t` seconds so a fast, turning robot doesn't
//! project a phantom barrier across the whole field).

use nalgebra::{Matrix2, Vector2};

use crate::types::Vec2;

/// Geometry of a single keep-out / keep-in region.
#[derive(Clone, Debug)]
pub enum ObstacleShape {
    /// Keep-out disk: clearance is `‖p − center‖ − radius` (safe when ≥ 0).
    Circle { center: Vec2, radius: f64 },
    /// Keep-in half-plane: clearance is `offset − normal·p` (safe when ≥ 0).
    /// `normal` is unit length and points toward the forbidden side.
    HalfPlane { normal: Vec2, offset: f64 },
    /// Axis-aligned keep-out box: clearance is the box signed distance —
    /// positive outside the box, negative inside.
    Box { min: Vec2, max: Vec2 },
}

/// A single soft obstacle: a shape, its barrier parameters, and an optional
/// constant-velocity drift used to extrapolate moving robots over the horizon.
#[derive(Clone, Debug)]
pub struct Obstacle {
    pub shape: ObstacleShape,
    /// Constant-velocity drift of the shape [mm/s]. Zero for static obstacles.
    pub vel: Vec2,
    /// Drift is clamped to this many seconds of look-ahead.
    pub vel_cap_t: f64,
    /// Barrier stiffness `w`.
    pub weight: f64,
    /// Influence distance `δ` [mm]: barrier is zero for clearance ≥ δ.
    pub influence: f64,
}

impl Obstacle {
    /// Static obstacle (no drift) with the given shape and barrier parameters.
    pub fn fixed(shape: ObstacleShape, weight: f64, influence: f64) -> Self {
        Self {
            shape,
            vel: Vec2::zeros(),
            vel_cap_t: 0.0,
            weight,
            influence,
        }
    }

    /// Clearance and its position gradient at `p`, with drift applied for time `t`.
    fn clearance_at(&self, p: Vec2, t: f64) -> (f64, Vec2) {
        // Translating the shape by `disp` is equivalent to evaluating the static
        // shape at `p − disp`; the position gradient is unchanged by translation.
        let disp = self.vel * t.clamp(0.0, self.vel_cap_t);
        self.shape.clearance(p - disp)
    }
}

impl ObstacleShape {
    /// Signed clearance `d` and its gradient `∂d/∂p` at position `p`. Clearance
    /// is positive in the safe region.
    fn clearance(&self, p: Vec2) -> (f64, Vec2) {
        match *self {
            ObstacleShape::Circle { center, radius } => {
                let delta = p - center;
                let dist = delta.norm();
                if dist > 1.0e-9 {
                    (dist - radius, delta / dist)
                } else {
                    // Exactly on the centre: pick an arbitrary outward direction.
                    (-radius, Vec2::new(1.0, 0.0))
                }
            }
            ObstacleShape::HalfPlane { normal, offset } => (offset - normal.dot(&p), -normal),
            ObstacleShape::Box { min, max } => {
                let center = 0.5 * (min + max);
                let half = 0.5 * (max - min);
                let rel = p - center;
                let sgn = Vec2::new(
                    if rel.x >= 0.0 { 1.0 } else { -1.0 },
                    if rel.y >= 0.0 { 1.0 } else { -1.0 },
                );
                // q = |rel| − half, per axis. q > 0 means outside along that axis.
                let q = Vec2::new(rel.x.abs() - half.x, rel.y.abs() - half.y);
                if q.x > 0.0 || q.y > 0.0 {
                    // Outside: Euclidean distance to the nearest face / corner.
                    let qpos = Vec2::new(q.x.max(0.0), q.y.max(0.0));
                    let dist = qpos.norm();
                    if dist > 1.0e-9 {
                        let grad = Vec2::new(qpos.x * sgn.x, qpos.y * sgn.y) / dist;
                        (dist, grad)
                    } else {
                        // Numerically on the boundary: push out along the axis
                        // that is (barely) outside.
                        if q.x >= q.y {
                            (q.x, Vec2::new(sgn.x, 0.0))
                        } else {
                            (q.y, Vec2::new(0.0, sgn.y))
                        }
                    }
                } else {
                    // Inside: clearance is the (negative) distance to the closest
                    // face; gradient points out along that axis.
                    if q.x >= q.y {
                        (q.x, Vec2::new(sgn.x, 0.0))
                    } else {
                        (q.y, Vec2::new(0.0, sgn.y))
                    }
                }
            }
        }
    }
}

/// Total obstacle penalty, its position gradient, and the Gauss-Newton position
/// Hessian at `(p, t)`.
pub fn penalty_derivs(obstacles: &[Obstacle], p: Vec2, t: f64) -> (f64, Vector2<f64>, Matrix2<f64>) {
    let mut cost = 0.0;
    let mut grad = Vector2::zeros();
    let mut hess = Matrix2::zeros();
    for ob in obstacles {
        let (d, dd) = ob.clearance_at(p, t);
        let slack = ob.influence - d;
        if slack > 0.0 {
            // P = ½ w slack²,  ∇P = −w slack ∇d,  H_GN = w ∇d ∇dᵀ.
            cost += 0.5 * ob.weight * slack * slack;
            grad -= ob.weight * slack * dd;
            hess += ob.weight * (dd * dd.transpose());
        }
    }
    (cost, grad, hess)
}

/// Scalar-only obstacle penalty (used by line-search rollouts).
pub fn penalty_scalar(obstacles: &[Obstacle], p: Vec2, t: f64) -> f64 {
    let mut cost = 0.0;
    for ob in obstacles {
        let (d, _) = ob.clearance_at(p, t);
        let slack = ob.influence - d;
        if slack > 0.0 {
            cost += 0.5 * ob.weight * slack * slack;
        }
    }
    cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn grad_matches_fd(obstacles: &[Obstacle], p: Vec2, t: f64) {
        let (_, g, _) = penalty_derivs(obstacles, p, t);
        let eps = 1.0e-3;
        for j in 0..2 {
            let mut pp = p;
            let mut pm = p;
            pp[j] += eps;
            pm[j] -= eps;
            let fd =
                (penalty_scalar(obstacles, pp, t) - penalty_scalar(obstacles, pm, t)) / (2.0 * eps);
            assert_abs_diff_eq!(g[j], fd, epsilon = 1.0e-4);
        }
    }

    #[test]
    fn circle_zero_outside_influence() {
        let obs = vec![Obstacle::fixed(
            ObstacleShape::Circle {
                center: Vec2::new(0.0, 0.0),
                radius: 180.0,
            },
            1.0,
            200.0,
        )];
        // 500 mm from centre, influence reaches to 380 mm → zero cost.
        assert_abs_diff_eq!(
            penalty_scalar(&obs, Vec2::new(500.0, 0.0), 0.0),
            0.0,
            epsilon = 1.0e-12
        );
        // Just inside the shell → positive cost and outward gradient.
        let p = Vec2::new(300.0, 0.0);
        assert!(penalty_scalar(&obs, p, 0.0) > 0.0);
        let (_, g, _) = penalty_derivs(&obs, p, 0.0);
        assert!(g.x < 0.0, "barrier should push back toward −x, got {}", g.x);
    }

    #[test]
    fn circle_gradient_matches_fd() {
        let obs = vec![Obstacle::fixed(
            ObstacleShape::Circle {
                center: Vec2::new(100.0, -50.0),
                radius: 180.0,
            },
            0.5,
            250.0,
        )];
        grad_matches_fd(&obs, Vec2::new(260.0, 30.0), 0.0);
    }

    #[test]
    fn halfplane_gradient_matches_fd() {
        // Keep-in wall at x = 1000: clearance = 1000 − x.
        let obs = vec![Obstacle::fixed(
            ObstacleShape::HalfPlane {
                normal: Vec2::new(1.0, 0.0),
                offset: 1000.0,
            },
            0.5,
            150.0,
        )];
        grad_matches_fd(&obs, Vec2::new(900.0, 0.0), 0.0);
        // Well inside → no cost.
        assert_abs_diff_eq!(
            penalty_scalar(&obs, Vec2::new(0.0, 0.0), 0.0),
            0.0,
            epsilon = 1.0e-12
        );
    }

    #[test]
    fn box_outside_face_gradient_matches_fd() {
        // Keep-out box [-100,100]², robot approaching the +x face from outside.
        let obs = vec![Obstacle::fixed(
            ObstacleShape::Box {
                min: Vec2::new(-100.0, -100.0),
                max: Vec2::new(100.0, 100.0),
            },
            0.5,
            200.0,
        )];
        grad_matches_fd(&obs, Vec2::new(230.0, 10.0), 0.0);
    }

    #[test]
    fn box_outside_corner_gradient_matches_fd() {
        let obs = vec![Obstacle::fixed(
            ObstacleShape::Box {
                min: Vec2::new(-100.0, -100.0),
                max: Vec2::new(100.0, 100.0),
            },
            0.5,
            300.0,
        )];
        // Off the +x/+y corner — both q components positive.
        grad_matches_fd(&obs, Vec2::new(200.0, 220.0), 0.0);
    }

    #[test]
    fn box_inside_pushes_out_nearest_face() {
        let obs = vec![Obstacle::fixed(
            ObstacleShape::Box {
                min: Vec2::new(-100.0, -100.0),
                max: Vec2::new(100.0, 100.0),
            },
            1.0,
            50.0,
        )];
        // Inside, closer to the +x face (x = 100) than any other.
        let p = Vec2::new(80.0, 5.0);
        assert!(penalty_scalar(&obs, p, 0.0) > 0.0);
        let (_, g, _) = penalty_derivs(&obs, p, 0.0);
        assert!(
            g.x < 0.0 && g.y.abs() < 1.0e-9,
            "inside box should push out along +x, got {:?}",
            g
        );
    }

    #[test]
    fn drift_shifts_circle_in_time() {
        // Circle drifting at +1000 mm/s in x, capped at 0.5 s.
        let obs = vec![Obstacle {
            shape: ObstacleShape::Circle {
                center: Vec2::new(0.0, 0.0),
                radius: 180.0,
            },
            vel: Vec2::new(1000.0, 0.0),
            vel_cap_t: 0.5,
            weight: 1.0,
            influence: 200.0,
        }];
        let p = Vec2::new(500.0, 0.0);
        // At t=0 the robot at x=500 is clear; by t=0.5 the circle centre reaches
        // x=500, so the robot is in deep penetration.
        assert_abs_diff_eq!(penalty_scalar(&obs, p, 0.0), 0.0, epsilon = 1.0e-12);
        assert!(penalty_scalar(&obs, p, 0.5) > 0.0);
        // Beyond the cap the obstacle stops advancing.
        assert_abs_diff_eq!(
            penalty_scalar(&obs, p, 1.0),
            penalty_scalar(&obs, p, 0.5),
            epsilon = 1.0e-9
        );
    }
}
