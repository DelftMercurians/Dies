use std::time::Duration;

use super::variable::Variable;

pub struct MTP<T: Variable> {
    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    setpoint: Option<T>,
    kp: f64,
    proportional_time_window: Duration,
}

impl<T: Variable> MTP<T> {
    pub fn new(max_accel: f64, max_speed: f64, max_decel: f64) -> Self {
        Self {
            max_accel,
            max_speed,
            max_decel,
            setpoint: None,
            kp: 1.0,
            proportional_time_window: Duration::from_millis(700),
        }
    }

    pub fn set_setpoint(&mut self, setpoint: T) {
        self.setpoint = Some(setpoint);
    }

    pub fn set_kp(&mut self, kp: f64) {
        self.kp = kp;
    }

    pub fn set_proportional_time_window(&mut self, window: Duration) {
        self.proportional_time_window = window;
    }

    pub fn update(&self, current: T, velocity: T, dt: f64) -> T {
        let setpoint = match self.setpoint {
            Some(s) => s,
            None => return T::zero(),
        };

        let displacement = setpoint - current;
        let distance = displacement.magnitude();
        let direction = if distance > f64::EPSILON {
            displacement * (1.0 / distance)
        } else {
            T::zero()
        };
        let current_speed = velocity.magnitude();

        // Overshoot detection
        if displacement.dot(velocity) < 0.0 {
            // Proportional control to reduce overshoot
            let proportional_velocity = direction * distance * self.kp;
            let dv = proportional_velocity - velocity;
            return velocity + dv.cap_magnitude(self.max_decel * dt);
        }

        // Calculate deceleration distance based on current_speed and max_decel
        let decel_distance = (self.max_speed * self.max_speed) / (2.0 * self.max_decel);

        let time_to_target = distance / current_speed;
        if time_to_target <= self.proportional_time_window.as_secs_f64() {
            // Proportional control
            let proportional_velocity = direction * distance * self.kp;
            let dv = proportional_velocity - velocity;
            velocity + dv.cap_magnitude(self.max_decel * dt)
        } else if distance <= decel_distance {
            // Deceleration phase
            let target_v = velocity
                - (velocity * (1.0 / (velocity.magnitude() + f64::EPSILON)) * self.max_decel * dt);
            let target_v = if (target_v * dt).magnitude() > distance {
                direction * distance / dt
            } else {
                target_v
            };
            let dv = target_v - velocity;
            velocity + dv.cap_magnitude(self.max_decel * dt)
        } else if current_speed < self.max_speed {
            // Acceleration phase
            let target_v = direction * self.max_speed;
            let dv = target_v - velocity;
            velocity + dv.cap_magnitude(self.max_accel * dt)
        } else {
            // Cruise phase
            velocity
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const DT: f64 = 1.0 / 60.0;

    #[test]
    fn test_mtp_update_without_setpoint() {
        let mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        assert_eq!(mtp.update(0.0, 0.0, DT), 0.0);
    }

    #[test]
    fn test_mtp_update_acceleration_phase() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let v1 = mtp.update(0.0, 0.0, DT);
        assert_relative_eq!(v1, 1.0 * DT, epsilon = 1e-6);
        let v2 = mtp.update(0.0, v1, DT);
        assert_relative_eq!(v2 - v1, 1.0 * DT, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_cruise_phase() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let v = mtp.update(0.0, 10.0, DT);
        assert_relative_eq!(v, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_deceleration_phase() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let v = mtp.update(95.0, 10.0, DT);
        assert!(v < 10.0, "Velocity should decrease ({v} < 10.0)");
        assert_relative_eq!(10.0 - v, 2.0 * DT, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_near_setpoint() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let v = mtp.update(99.9, 0.1, DT);
        assert!(v < 0.1, "Velocity should decrease ({v} < 0.1)");
        assert_relative_eq!(0.1 - v, 2.0 * DT, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_at_setpoint() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        assert_relative_eq!(mtp.update(100.0, 0.0, DT), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_overshoot() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let v = mtp.update(101.0, 1.0, DT);
        assert!(v < 1.0);
        assert_relative_eq!(1.0 - v, 2.0 * DT, epsilon = 1e-6);
    }

    #[test]
    fn test_mtp_update_multiple_steps() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        let mut position = 0.0;
        let mut velocity = 0.0;
        for _ in 0..60 {
            // Simulate 1 second
            let new_velocity = mtp.update(position, velocity, DT);
            assert!(
                (new_velocity - velocity).abs() <= 1.0 * DT + 1e-6,
                "Velocity change exceeds maximum acceleration"
            );
            position += (velocity + new_velocity) * 0.5 * DT; // Trapezoidal integration
            velocity = new_velocity;
        }
        assert!(
            position > 0.0 && position < 100.0,
            "Position after 1 second should be between 0 and 100"
        );
        assert!(
            velocity > 0.0 && velocity <= 10.0,
            "Velocity after 1 second should be between 0 and 10"
        );
    }

    #[test]
    fn test_velocity_change_limit() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 1.0);
        mtp.set_setpoint(100.0);
        for initial_velocity in [0.0, 5.0, 10.0] {
            let new_velocity = mtp.update(0.0, initial_velocity, DT);
            assert!(
                (new_velocity - initial_velocity).abs() <= 1.0 * DT + 1e-6,
                "Velocity change exceeds maximum acceleration ({initial_velocity} -> {new_velocity})"
            );
        }
    }
}