//! Bounded motion regions with a no-overshoot velocity envelope.
//!
//! A [`MotionBounds`] describes a region a robot must stay inside. Its core
//! operation, [`MotionBounds::clamp_velocity`], caps the component of a commanded
//! velocity that points *out of* the region so the robot can always brake to a
//! stop before crossing the boundary, given its deceleration limit
//! (`v² ≤ 2·a·d`). Motion *along* a boundary is left untouched.
//!
//! This makes overshoot past an edge provably impossible regardless of how
//! aggressive the upstream controller is, and — because it keys off the robot's
//! *position* rather than its (lag-prone) velocity estimate — it is robust to the
//! velocity-estimation lag in the tracker.
//!
//! Coordinates are team-relative millimetres, the same frame the player
//! controllers operate in.

use serde::{Deserialize, Serialize};

use crate::Vector2;

/// A bounded region for a robot's motion. Extend with more shapes (e.g. `Rect`)
/// as needed; each implements the same braking-envelope contract.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum MotionBounds {
    /// An annular sector around a centre — a "front of goal" arc band.
    Arc(ArcZone),
}

/// An annular sector around `center`, opening along the `+x` axis. The keeper's
/// guard region: a band between two radii, spanning `±half_angle` off straight-out.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ArcZone {
    /// Centre of the arc (own-goal centre, team-relative).
    pub center: Vector2,
    /// Inner radius — the robot may not come closer to `center` than this.
    pub min_radius: f64,
    /// Outer radius — the hard "don't charge out" bound.
    pub max_radius: f64,
    /// Maximum angular excursion off the `+x` axis (radians), each side.
    pub half_angle: f64,
}

impl MotionBounds {
    /// Whether `pos` lies inside the region.
    pub fn contains(&self, pos: Vector2) -> bool {
        match self {
            MotionBounds::Arc(z) => z.contains(pos),
        }
    }

    /// Clamp `vel` so the robot can brake before leaving the region.
    ///
    /// Only the component of `vel` pointing out of the region is limited — to
    /// `sqrt(2·decel·max(d − margin, 0))`, where `d` is the distance to that
    /// boundary. Motion along a boundary is unchanged. `margin` (mm) pulls the
    /// effective edge in slightly to absorb position noise and one-tick
    /// discretization. `decel` must match the deceleration the controller will
    /// actually apply, or the guarantee is only approximate.
    pub fn clamp_velocity(&self, pos: Vector2, vel: Vector2, decel: f64, margin: f64) -> Vector2 {
        match self {
            MotionBounds::Arc(z) => z.clamp_velocity(pos, vel, decel, margin),
        }
    }
}

impl ArcZone {
    fn contains(&self, pos: Vector2) -> bool {
        let rel = pos - self.center;
        let r = rel.norm();
        let theta = rel.y.atan2(rel.x);
        r >= self.min_radius && r <= self.max_radius && theta.abs() <= self.half_angle
    }

    fn clamp_velocity(&self, pos: Vector2, vel: Vector2, decel: f64, margin: f64) -> Vector2 {
        let rel = pos - self.center;
        let r = rel.norm();
        // At the centre the radial/tangential frame is undefined; nothing useful
        // to clamp (and the keeper never sits on the goal centre).
        if r < 1.0e-6 {
            return vel;
        }
        let radial = rel / r; // outward (+r)
        let tangent = Vector2::new(-radial.y, radial.x); // +θ (CCW)
        let theta = rel.y.atan2(rel.x);

        // Max speed that can still brake to a stop within distance `d`.
        let brake = |d: f64| (2.0 * decel.max(0.0) * (d - margin).max(0.0)).sqrt();

        let mut v_r = vel.dot(&radial);
        let mut v_t = vel.dot(&tangent);

        // Radial: outward → bounded by the outer radius; inward → by the inner.
        if v_r > 0.0 {
            v_r = v_r.min(brake(self.max_radius - r));
        } else if v_r < 0.0 {
            v_r = v_r.max(-brake(r - self.min_radius));
        }

        // Tangential: +θ heads toward +half_angle, −θ toward −half_angle. The
        // remaining travel is an arc length (angle × radius).
        if v_t > 0.0 {
            v_t = v_t.min(brake((self.half_angle - theta).max(0.0) * r));
        } else if v_t < 0.0 {
            v_t = v_t.max(-brake((self.half_angle + theta).max(0.0) * r));
        }

        radial * v_r + tangent * v_t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zone() -> MotionBounds {
        MotionBounds::Arc(ArcZone {
            center: Vector2::new(-4500.0, 0.0),
            min_radius: 0.0,
            max_radius: 400.0,
            half_angle: std::f64::consts::FRAC_PI_4,
        })
    }

    #[test]
    fn outward_radial_speed_capped_to_brake_envelope() {
        let z = zone();
        let decel = 8000.0;
        // On the +x axis, 100mm inside the outer radius.
        let pos = Vector2::new(-4500.0 + 300.0, 0.0); // r = 300, d = 100
        let v = z.clamp_velocity(pos, Vector2::new(5000.0, 0.0), decel, 0.0);
        let expected = (2.0 * decel * 100.0).sqrt();
        assert!((v.x - expected).abs() < 1.0, "got {:?}, want {expected}", v);
        assert!(v.y.abs() < 1.0e-9);
    }

    #[test]
    fn stops_exactly_at_the_outer_edge() {
        let z = zone();
        let pos = Vector2::new(-4500.0 + 400.0, 0.0); // r = max_radius
        let v = z.clamp_velocity(pos, Vector2::new(5000.0, 0.0), 8000.0, 0.0);
        assert!(v.norm() < 1.0e-9, "should be fully stopped: {:?}", v);
    }

    #[test]
    fn inward_motion_is_unrestricted_far_from_inner() {
        let z = zone();
        let pos = Vector2::new(-4500.0 + 400.0, 0.0); // at outer edge, r = 400
                                                      // Inward (−x) speed within the inner-radius envelope (brake(400) ≈ 2530).
        let v = z.clamp_velocity(pos, Vector2::new(-2000.0, 0.0), 8000.0, 0.0);
        assert!((v.x - (-2000.0)).abs() < 1.0, "inward unchanged: {:?}", v);
    }

    #[test]
    fn tangential_along_boundary_unchanged_mid_arc() {
        let z = zone();
        // On the +x axis (θ=0, far from the ±45° ends), pure tangential motion.
        let pos = Vector2::new(-4500.0 + 400.0, 0.0);
        let v = z.clamp_velocity(pos, Vector2::new(0.0, 2000.0), 8000.0, 0.0);
        assert!((v.y - 2000.0).abs() < 1.0, "tangential unchanged: {:?}", v);
        assert!(v.x.abs() < 1.0e-6);
    }

    #[test]
    fn tangential_capped_near_angular_end() {
        let z = zone();
        // Just inside the +45° end: small remaining arc → tangential speed capped.
        let r = 400.0;
        let theta = std::f64::consts::FRAC_PI_4 - 0.05; // ~2.9° from the end
        let pos = Vector2::new(-4500.0 + r * theta.cos(), r * theta.sin());
        let fast = z.clamp_velocity(pos, Vector2::new(-3000.0, 3000.0), 8000.0, 0.0);
        let v_t = fast.dot(&Vector2::new(-theta.sin(), theta.cos()));
        let d_arc = 0.05 * r;
        assert!(v_t <= (2.0 * 8000.0 * d_arc).sqrt() + 1.0, "capped: {v_t}");
    }

    #[test]
    fn contains_respects_radius_and_angle() {
        let z = zone();
        assert!(z.contains(Vector2::new(-4500.0 + 300.0, 0.0)));
        assert!(!z.contains(Vector2::new(-4500.0 + 500.0, 0.0))); // too far out
                                                                  // Beyond the +45° clamp at full radius.
        let theta = std::f64::consts::FRAC_PI_4 + 0.2;
        assert!(!z.contains(Vector2::new(
            -4500.0 + 400.0 * theta.cos(),
            400.0 * theta.sin()
        )));
    }
}
