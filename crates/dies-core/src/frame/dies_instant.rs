use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A point in time represented as a number of seconds since the start of the application.
///
/// This instant is guaranteed to be:
///  - monotonically increasing
///  - non-negative
///  - finite and non-NaN
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
pub struct DiesInstant(f64);

impl DiesInstant {
    fn new(value: f64) -> Self {
        assert!(value >= 0.0, "DiesInstant must be non-negative");
        assert!(value.is_finite(), "DiesInstant must be finite");
        Self(value)
    }

    /// Get the underlying floating point value.
    pub fn as_secs_f64(&self) -> f64 {
        self.0
    }

    /// Get the duration between this instant and another instant.
    ///
    /// If the other instant is after this instant, the result is 0, therefore this
    /// value is guaranteed to be non-negative.
    pub fn duration_since(&self, other: &Self) -> f64 {
        if self.0 < other.0 {
            return 0.0;
        }
        self.0 - other.0
    }

    pub fn test_value(value: f64) -> Self {
        Self::new(value)
    }
}

impl std::ops::Add<f64> for DiesInstant {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        assert!(rhs >= 0.0, "DiesInstant cannot decrease");
        Self::new(self.0 + rhs)
    }
}

impl std::ops::Add<Duration> for DiesInstant {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        Self::new(self.0 + rhs.as_secs_f64())
    }
}

impl std::ops::AddAssign<f64> for DiesInstant {
    fn add_assign(&mut self, rhs: f64) {
        assert!(rhs >= 0.0, "DiesInstant cannot decrease");
        *self = Self::new(self.0 + rhs);
    }
}

impl std::ops::AddAssign<Duration> for DiesInstant {
    fn add_assign(&mut self, rhs: Duration) {
        *self = Self::new(self.0 + rhs.as_secs_f64());
    }
}
