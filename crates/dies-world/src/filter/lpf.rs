use std::f64::consts::PI;

/// A low-pass filter for angular values (in radians).
pub struct AngleLowPassFilter {
    alpha: f64,
    filtered_angle: Option<f64>,
}

impl AngleLowPassFilter {
    /// Creates a new `AngleLowPassFilter` with the specified alpha value.
    pub fn new(alpha: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "Alpha must be between 0 and 1"
        );
        AngleLowPassFilter {
            alpha,
            filtered_angle: None,
        }
    }

    /// Updates the filter with a new angle and returns the filtered result.
    pub fn update(&mut self, angle: f64) -> f64 {
        let normalized_angle = Self::normalize_angle(angle);

        if let Some(filtered) = self.filtered_angle {
            // Calculate the difference, handling wraparound
            let mut diff = normalized_angle - filtered;
            if diff > PI {
                diff -= 2.0 * PI;
            } else if diff < -PI {
                diff += 2.0 * PI;
            }

            // Apply the filter
            let new_filtered = filtered + self.alpha * diff;
            self.filtered_angle = Some(Self::normalize_angle(new_filtered));
        } else {
            self.filtered_angle = Some(normalized_angle);
        }

        self.filtered_angle.unwrap()
    }

    /// Normalizes an angle to be within the range [-π, π].
    fn normalize_angle(angle: f64) -> f64 {
        let mut normalized = angle % (2.0 * PI);
        if normalized > PI {
            normalized -= 2.0 * PI;
        } else if normalized < -PI {
            normalized += 2.0 * PI;
        }
        normalized
    }

    pub fn update_settings(&mut self, alpha: f64) {
        self.alpha = alpha;
    }
}
