use std::time::Instant;

use super::variable::Variable;

pub struct MTP<T> {
    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    decel_distance: Option<f64>,
    setpoint: Option<T>,
}

impl<T> MTP<T>
where
    T: Variable,
{
    pub fn new(max_accel: f64, max_speed: f64, max_decel: f64) -> Self {
        Self {
            max_accel,
            max_speed,
            max_decel,
            setpoint: None,
            decel_distance: None,
        }
    }

    pub fn set_setpoint(&mut self, setpoint: T) {
        self.setpoint = Some(setpoint);
    }

    pub fn set_decel_distance(&mut self, decel_distance: f64) {
        self.decel_distance = Some(decel_distance);
    }

    pub fn update(&mut self, current: T, derivative: T, dt: f64) -> T {
        if self.setpoint.is_none() || self.decel_distance.is_none() {
            return T::zero();
        }

        let point = self.setpoint.unwrap();
        let decel_distance = self.decel_distance.unwrap();

        let future_dir = point - current;
        let diff = derivative - future_dir;

        let future_velocity: T;

        if future_dir.magnitude() < decel_distance {
            future_velocity = derivative - (future_dir * self.max_decel);
        } else {
            future_velocity = derivative + (future_dir * self.max_accel);
        }

        if future_velocity.magnitude() < 0.0 {
            return T::zero();
        } else if future_velocity.magnitude() > self.max_speed {
            return future_dir * (future_velocity.magnitude() / self.max_speed);
        } else {
            return future_velocity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn test_mtp_update_without_setpoint() {
        let mut pid = MTP::<f64>::new(1.0, 1.0, 1.0);

        let output = pid.update(1.0);

        assert_eq!(output, 0.0);
    }

    #[test]
    fn test_with_float() {
        let mut mtp = MTP::<f64>::new()
    }
}
