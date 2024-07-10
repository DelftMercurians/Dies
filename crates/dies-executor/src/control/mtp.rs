use std::{fmt::Debug, time::Duration};

use dies_core::variable::Variable;

pub struct MTP<T: Variable> {
    max_accel: f64,
    max_speed: f64,
    max_decel: f64,
    setpoint: Option<T>,
    kp: f64,
    proportional_time_window: Duration,
    cutoff_distance: f64,
}

impl<T: Variable + Debug> MTP<T> {
    pub fn new(max_accel: f64, max_speed: f64, max_decel: f64) -> Self {
        Self {
            max_accel,
            max_speed,
            max_decel,
            setpoint: None,
            kp: 1.0,
            proportional_time_window: Duration::from_millis(700),
            cutoff_distance: 0.,
        }
    }

    pub fn set_setpoint(&mut self, setpoint: T) {
        self.setpoint = Some(setpoint);
    }

    pub fn max_accel(&self) -> f64 {
        self.max_accel
    }
    pub fn max_speed(&self) -> f64 {
        self.max_speed
    }
    pub fn max_decel(&self) -> f64 {
        self.max_decel
    }
    pub fn setpoint(&self) -> Option<T> {
        self.setpoint
    }
    pub fn kp(&self) -> f64 {
        self.kp
    }
    pub fn proportional_time_window(&self) -> Duration {
        self.proportional_time_window
    }
    pub fn cutoff_distance(&self) -> f64 {
        self.cutoff_distance
    }

    pub fn update_settings(
        &mut self,
        max_accel: f64,
        max_speed: f64,
        max_decel: f64,
        kp: f64,
        proportional_time_window: Duration,
        cutoff_distance: f64,
    ) {
        self.max_accel = max_accel;
        self.max_speed = max_speed;
        self.max_decel = max_decel;
        self.kp = kp;
        self.proportional_time_window = proportional_time_window;
        self.cutoff_distance = cutoff_distance;
    }

    pub fn update(&self, current: T, velocity: T, dt: f64) -> T {
        let setpoint = match self.setpoint {
            Some(s) => s,
            None => return T::zero(),
        };

        let displacement = setpoint - current;
        let distance = displacement.magnitude();

        if distance < self.cutoff_distance {
            // println!("Sending cutoff");

            return T::zero();
        }

        // compute the normalized direction with the displacement and the distance
        // the if condition statement prevents a division by 0
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
            // println!("Overshoot case returning");
            // dies_core::debug_string("p5.Goal", format!("{:?}", velocity + dv.cap_magnitude(self.max_decel * dt)));

            // besides decreasing the velocity with dv, we also go in the opposite direction to compensate for the overshoot

            dies_core::debug_string("p5.MTPMode", "Overshoot");
            return -velocity + dv.cap_magnitude(self.max_decel * dt);
        }

        // Calculate deceleration distance based on current_speed and max_decel
        let decel_distance = (current_speed * current_speed) / (2.0 * self.max_decel);

        let time_to_target = distance / current_speed;
        if time_to_target <= self.proportional_time_window.as_secs_f64() {
            // Proportional control
            let proportional_velocity_magnitude = distance * self.kp;
            let dv_magnitude = proportional_velocity_magnitude - velocity.magnitude();

            dies_core::debug_string("p5.MTPMode", "Proportional");
            let new_speed = current_speed + dv_magnitude.cap_magnitude(self.max_decel * dt);
            direction * new_speed
        } else if distance <= decel_distance && current_speed > 0.0 {
            // Deceleration phase
            let target_v = velocity
                - (velocity * (1.0 / (velocity.magnitude() + f64::EPSILON)) * self.max_decel * dt);
            let target_v = if (target_v * dt).magnitude() < distance {
                direction * distance / dt
            } else {
                target_v
            };
            let dv_mag = target_v.magnitude() - velocity.magnitude();
            dies_core::debug_string("p5.MTPMode", "Deceleration");
            let new_speed = current_speed + dv_mag.min(self.max_accel * dt);
            direction * new_speed
        } else if current_speed < self.max_speed {
            // Acceleration phase
            let target_v = self.max_speed;
            let dv = target_v - current_speed;
            let new_speed = current_speed + dv.min(self.max_accel * dt);
            dies_core::debug_string("p5.MTPMode", "Acceleration");
            direction * new_speed
        } else {
            // Cruise phase
            dies_core::debug_string("p5.MTPMode", "Cruise");
            direction * velocity.magnitude()
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
    fn test_zero_velocity() {
        let mut mtp = MTP::<f64>::new(1.0, 10.0, 2.0);
        mtp.set_setpoint(100.0);
        assert_eq!(mtp.update(0.0, 0.0, DT), 1.0 * DT);
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
