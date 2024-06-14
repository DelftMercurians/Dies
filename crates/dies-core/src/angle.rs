use serde::{Deserialize, Serialize};

use crate::Vector2;

/// An angle in radians, always in (-pi, pi]. This type supports safe arithmetic
/// operations:
///
/// ```no_run
/// # use dies_core::Angle;
/// let a = Angle::from_degrees(90.0);
/// let b = Angle::from_degrees(45.0);
/// let c = a + b;
/// assert_eq!(c.degrees(), 135.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Angle(f64);

impl Angle {
    /// Create a new angle from radians.
    pub fn from_radians(radians: f64) -> Self {
        Angle(wrap_angle(radians))
    }

    /// Create a new angle from degrees.
    pub fn from_degrees(degrees: f64) -> Self {
        Self::from_radians(degrees.to_radians())
    }

    /// Get the angle in radians.
    pub fn radians(&self) -> f64 {
        self.0
    }

    /// Get the angle in degrees.
    pub fn degrees(&self) -> f64 {
        self.0.to_degrees()
    }

    /// Rotate a vector by this angle.
    pub fn rotate_vector(&self, v: Vector2) -> Vector2 {
        let rot = nalgebra::Rotation2::new(self.0);
        rot * v
    }
}

impl std::ops::Add for Angle {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Angle::from_radians(self.0 + other.0)
    }
}

impl std::ops::Sub for Angle {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Angle::from_radians(self.0 - other.0)
    }
}

impl std::ops::Neg for Angle {
    type Output = Self;

    fn neg(self) -> Self {
        Angle::from_radians(-self.0)
    }
}

impl std::ops::AddAssign for Angle {
    fn add_assign(&mut self, other: Self) {
        self.0 = wrap_angle(self.0 + other.0);
    }
}

impl std::ops::SubAssign for Angle {
    fn sub_assign(&mut self, other: Self) {
        self.0 = wrap_angle(self.0 - other.0);
    }
}

impl std::ops::Mul<Vector2> for Angle {
    type Output = Vector2;

    fn mul(self, v: Vector2) -> Vector2 {
        self.rotate_vector(v)
    }
}

impl std::fmt::Display for Angle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} rad", self.0)
    }
}

fn wrap_angle(angle: f64) -> f64 {
    let mut angle = angle % (2.0 * std::f64::consts::PI);
    if angle <= -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    } else if angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_wrap_angle() {
        assert_eq!(wrap_angle(0.0), 0.0);
        assert_eq!(wrap_angle(PI), PI);
        assert_eq!(wrap_angle(-PI), PI);
        assert_eq!(wrap_angle(3.0 * PI), PI);
        assert_eq!(wrap_angle(-3.0 * PI), PI);
    }

    #[test]
    fn test_angle_add() {
        let a = Angle::from_degrees(90.0);
        let b = Angle::from_degrees(45.0);
        let c = a + b;
        assert_eq!(c.degrees(), 135.0);
    }

    #[test]
    fn test_angle_sub() {
        let a = Angle::from_degrees(90.0);
        let b = Angle::from_degrees(45.0);
        let c = a - b;
        assert_eq!(c.degrees(), 45.0);
    }

    #[test]
    fn test_angle_neg() {
        let a = Angle::from_degrees(90.0);
        let b = -a;
        assert_eq!(b.degrees(), -90.0);
    }

    #[test]
    fn test_angle_add_assign() {
        let mut a = Angle::from_degrees(90.0);
        let b = Angle::from_degrees(45.0);
        a += b;
        assert_eq!(a.degrees(), 135.0);
    }

    #[test]
    fn test_angle_sub_assign() {
        let mut a = Angle::from_degrees(90.0);
        let b = Angle::from_degrees(45.0);
        a -= b;
        assert_eq!(a.degrees(), 45.0);
    }

    #[test]
    fn test_angle_mul() {
        let a = Angle::from_degrees(90.0);
        let v = Vector2::new(1.0, 0.0);
        let r = a * v;
        assert_eq!(r.y, 1.0);
    }
}
