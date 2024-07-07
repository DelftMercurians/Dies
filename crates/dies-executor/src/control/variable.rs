use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use dies_core::{Vector2, Vector3};

/// A trait for types that can be used as variables in a PID controller.
pub trait Variable:
    Copy
    + Add<Self, Output = Self>
    + AddAssign
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + Sub<Output = Self>
    + PartialOrd
    + Neg<Output = Self>
{
    /// Returns the zero value for this type.
    fn zero() -> Self;

    /// Returns the magnitude of this variable. This is always a positive value.
    fn magnitude(self) -> f64;

    /// Returns the dot product of this variable with another.
    fn dot(self, rhs: Self) -> f64;

    fn cap_magnitude(self, max: f64) -> Self {
        let magnitude = self.magnitude();
        if magnitude > max {
            self * (max / magnitude)
        } else {
            self
        }
    }
}

impl Variable for f64 {
    fn zero() -> Self {
        0.0
    }

    fn magnitude(self) -> f64 {
        self.abs()
    }

    fn dot(self, rhs: Self) -> f64 {
        self * rhs
    }
}

impl Variable for Vector2 {
    fn zero() -> Self {
        Vector2::zeros()
    }

    fn magnitude(self) -> f64 {
        self.norm()
    }

    fn dot(self, rhs: Self) -> f64 {
        Vector2::dot(&self, &rhs)
    }

    fn cap_magnitude(self, max: f64) -> Self {
        self.simd_cap_magnitude(max)
    }
}

impl Variable for Vector3 {
    fn zero() -> Self {
        Vector3::zeros()
    }

    fn magnitude(self) -> f64 {
        self.norm()
    }

    fn dot(self, rhs: Self) -> f64 {
        Vector3::dot(&self, &rhs)
    }

    fn cap_magnitude(self, max: f64) -> Self {
        self.simd_cap_magnitude(max)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_cap_magnitude_below_max() {
        let value = 5.0;
        let max = 10.0;
        let capped_value = value.cap_magnitude(max);
        assert_eq!(capped_value, value);
    }

    #[test]
    fn test_cap_magnitude_above_max() {
        let value = 15.0;
        let max = 10.0;
        let capped_value = value.cap_magnitude(max);
        assert_relative_eq!(capped_value, 10.0);
    }

    #[test]
    fn test_negative_cap_magnitude_below_max() {
        let value = -15.0;
        let max = 10.0;
        let capped_value = value.cap_magnitude(max);
        assert_relative_eq!(capped_value, -10.0);
    }
}
