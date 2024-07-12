use std::f64::consts::PI;

use serde::{Deserialize, Serialize};
use typeshare::typeshare;

use crate::Vector2;

/// An angle in radians, always in (-pi, pi]. This type supports safe arithmetic
/// operations:
///
/// ```ignore
/// # use dies_core::Angle;
/// let a = Angle::from_degrees(90.0);
/// let b = Angle::from_degrees(45.0);
/// let c = a + b;
/// assert_eq!(c.degrees(), 135.0);
/// ```
#[derive(Debug, Clone, Copy, PartialOrd, Serialize, Deserialize)]
#[typeshare(serialized_as = "f64")]
pub struct Angle(f64);

impl Angle {
    pub const TWO_PI: Angle = Angle(2.0 * PI);
    pub const PI: Angle = Angle(PI);
    pub const PI_2: Angle = Angle(PI / 2.0);
    pub const PI_4: Angle = Angle(PI / 4.0);

    /// Create a new angle from radians.
    pub fn from_radians(radians: f64) -> Self {
        Angle(wrap_angle(radians))
    }

    /// Create a new angle from degrees.
    pub fn from_degrees(degrees: f64) -> Self {
        Self::from_radians(degrees.to_radians())
    }

    /// Compute the smallest signed counter-clockwise angle from point a to point b.
    pub fn between_points(a: Vector2, b: Vector2) -> Self {
        let angle = (b.y - a.y).atan2(b.x - a.x);
        Self::from_radians(angle)
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
    pub fn rotate_vector(&self, v: &Vector2) -> Vector2 {
        let rot = nalgebra::Rotation2::new(self.0);
        rot * v
    }

    /// Get the inversion of the angle (* -1)
    pub fn inv(&self) -> Self {
        -*self
    }

    /// Get a random angle in (-pi, pi]
    pub fn random() -> Self {
        let radians = (rand::random::<f64>() * 2.0 * PI) - PI;
        Self::from_radians(radians)
    }

    /// Get the absolute value of the angle
    pub fn abs(&self) -> f64 {
        self.0.abs()
    }

    /// Get the sign of the angle
    pub fn signum(&self) -> f64 {
        self.0.signum()
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
        self.rotate_vector(&v)
    }
}

impl std::ops::Mul<f64> for Angle {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self {
        Angle::from_radians(self.0 * scalar)
    }
}

impl std::ops::Div<f64> for Angle {
    type Output = Self;

    fn div(self, scalar: f64) -> Self {
        Angle::from_radians(self.0 / scalar)
    }
}

impl std::fmt::Display for Angle {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} rad", self.0)
    }
}

impl Default for Angle {
    fn default() -> Self {
        Self::from_radians(0.0)
    }
}

impl PartialEq for Angle {
    fn eq(&self, other: &Self) -> bool {
        let diff: f64 = (self.radians() - other.radians()).abs();
        const TOLERANCE: f64 = 1e-5; // about sqrt of f32 precision
        !(TOLERANCE..=(2.0 * PI - TOLERANCE)).contains(&diff)
    }
}

fn wrap_angle(angle: f64) -> f64 {
    let mut angle = angle % (2.0 * PI);
    if angle <= -PI {
        angle += 2.0 * PI;
    } else if angle > PI {
        angle -= 2.0 * PI;
    }
    angle
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

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
    fn between_points() {
        let a = Vector2::new(0.0, 0.0);
        let b = Vector2::new(1.0, 1.0);
        let angle = Angle::between_points(a, b);
        assert_eq!(angle.degrees(), 45.0);
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

        let a = Angle::from_degrees(-180.0);
        let b = Angle::from_degrees(180.0);
        assert_eq!((a - b).degrees(), 0.0);

        let a = Angle::from_degrees(180.0);
        let b = Angle::from_degrees(-179.0);
        assert_relative_eq!((a - b).degrees(), -1.0, epsilon = 1e-5);
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

    #[test]
    fn test_flip_around_y_axis() {
        fn flip(angle: Angle) -> Angle {
            Angle::PI + angle
        }

        let a = Angle::from_degrees(90.0);
        assert_relative_eq!(flip(a).degrees(), -90.0, epsilon = 1e-5);

        let b = Angle::from_degrees(-90.0);
        assert_relative_eq!(flip(b).degrees(), 90.0, epsilon = 1e-5);

        let c = Angle::from_degrees(0.0);
        assert_relative_eq!(flip(c).degrees(), 180.0, epsilon = 1e-5);

        let c = Angle::from_degrees(180.0);
        assert_relative_eq!(flip(c).degrees(), 0.0, epsilon = 1e-5);
        let c = Angle::from_degrees(-180.0);
        assert_relative_eq!(flip(c).degrees(), 0.0, epsilon = 1e-5);

        let c = Angle::from_degrees(45.0);
        assert_relative_eq!(flip(c).degrees(), -135.0, epsilon = 1e-5);
    }

    #[test]
    fn test_angle_between_points() {
        let a = Vector2::new(0.0, 0.0);
        let b = Vector2::new(1.0, 1.0);
        let angle1 = Angle::between_points(a, b);
        assert_eq!(angle1.degrees(), 45.0);

        let angle2 = Angle::between_points(b, a);
        assert_eq!(angle2.degrees(), -135.0);
    }
}
