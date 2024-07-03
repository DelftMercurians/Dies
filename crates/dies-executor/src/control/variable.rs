use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use dies_core::{Vector2, Vector3};

/// A trait for types that can be used as variables in a PID controller.
pub trait Variable:
    Copy
    + Add<Self, Output = Self>
    + AddAssign
    + Mul<f64, Output = Self>
    + Sub<Output = Self>
    + Div<f64, Output = Self>
    + PartialOrd
    + Neg<Output = Self>
{
    /// Returns the zero value for this type.
    fn zero() -> Self;

    fn magnitude(self) -> f64;
}

impl Variable for f64 {
    fn zero() -> Self {
        0.0
    }

    fn magnitude(self) -> f64 {
        self
    }
}

impl Variable for Vector2 {
    fn zero() -> Self {
        Vector2::zeros()
    }

    fn magnitude(self) -> f64 {
        self.magnitude_squared()
    }
}

impl Variable for Vector3 {
    fn zero() -> Self {
        Vector3::zeros()
    }

    fn magnitude(self) -> f64 {
        self.magnitude_squared()
    }
}
