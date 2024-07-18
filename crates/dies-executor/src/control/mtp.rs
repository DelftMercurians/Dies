use std::time::Duration;

use dies_core::Vector2;

pub struct MTP {
    setpoint: Option<Vector2>,
    kp: f64,
    proportional_time_window: Duration,
    cutoff_distance: f64,
}

impl MTP {
    pub fn new() -> Self {
        Self { // this parameters seemingly don't matter, since stuff is loaded from config
            setpoint: None,
            kp: 1.0,
            proportional_time_window: Duration::from_millis(700),
            cutoff_distance: 30.0
        }
    }

    pub fn set_setpoint(&mut self, setpoint: Vector2) {
        self.setpoint = Some(setpoint);
    }

    pub fn deceleration_window(&self, velocity: Vector2, max_decel: f64) -> f64 {
        let current_speed = velocity.magnitude();
        (current_speed * current_speed) / (2.0 * max_decel)
    }

    pub fn update_settings(
        &mut self,
        kp: f64,
        proportional_time_window: Duration,
        cutoff_distance: f64,
    ) {
        self.kp = kp;
        self.proportional_time_window = proportional_time_window;
        self.cutoff_distance = cutoff_distance;
    }

    pub fn update(
        &self,
        current: Vector2,
        velocity: Vector2,
        dt: f64,
        max_accel: f64,
        max_speed: f64,
        max_decel: f64,
        carefullness: f64,
    ) -> Vector2 {
        let setpoint = match self.setpoint {
            Some(s) => s,
            None => return Vector2::zeros(),
        };

        let displacement = setpoint - current;
        let distance = displacement.magnitude();

        if distance < self.cutoff_distance {

            return Vector2::zeros();
        }

        // compute the normalized direction with the displacement and the distance
        // the if condition statement prevents a division by 0
        let direction = if distance > f64::EPSILON {
            displacement * (1.0 / distance)
        } else {
            Vector2::zeros()
        };
        let current_speed = velocity.magnitude();

        // Overshoot detection
        // if displacement.dot(&velocity) < 0.0 {
        //     // Proportional control to reduce overshoot
        //     let proportional_velocity = direction * distance * self.kp;
        //     let dv = proportional_velocity - velocity;
        //     // println!("Overshoot case returning");
        //     // dies_core::debug_string("p5.Goal", format!("{:?}", velocity + dv.cap_magnitude(self.max_decel * dt)));

        //     // besides decreasing the velocity with dv, we also go in the opposite direction to compensate for the overshoot

        //     dies_core::debug_string("p5.MTPMode", "Overshoot");
        //     return -velocity + dv.cap_magnitude(self.max_decel * dt);
        // }

        let care_factor = carefullness * 0.2;
        let time_to_target = distance / max_speed;
        if time_to_target <= self.proportional_time_window.as_secs_f64() {
            // Proportional control
            let proportional_velocity_magnitude = f64::max(distance - self.cutoff_distance, 0.0) * (self.kp * (1.0 - care_factor)) + 100.0 * care_factor;
            let dv_magnitude = proportional_velocity_magnitude - velocity.magnitude();

            let mut v_control = dv_magnitude;
            if v_control < 0.0 { // decelerate a lot faster than accelerate since inertia
                v_control *= 5.0 * (1.0 + care_factor);
            }

            dies_core::debug_string("p5.MTPMode", "Proportional");
            let new_speed = current_speed + cap_magnitude(v_control, max_decel * dt);
            direction * new_speed
        } else if current_speed < max_speed {
            // Acceleration phase
            dies_core::debug_string("p5.MTPMode", "Acceleration");
            let new_speed = current_speed + max_accel * dt;
            direction * new_speed
        } else {
            // Cruise phase
            dies_core::debug_string("p5.MTPMode", "Cruise");
            direction * max_speed
        }
    }
}

fn cap_magnitude(v: f64, max: f64) -> f64 {
    if v > max {
        max
    } else if v < -max {
        -max
    } else {
        v
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_relative_eq;

//     const DT: f64 = 1.0 / 60.0;

//     #[test]
//     fn test_mtp_update_without_setpoint() {
//         let mtp = MTP::new(1.0, 10.0, 2.0);
//         assert_eq!(
//             mtp.update(Vector2::zeros(), Vector2::zeros(), DT),
//             Vector2::zeros()
//         );
//     }

//     #[test]
//     fn test_zero_velocity() {
//         let mut mtp = MTP::new(1.0, 10.0, 2.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         let result = mtp.update(Vector2::zeros(), Vector2::zeros(), DT);
//         assert_relative_eq!(result.magnitude(), 1.0 * DT, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_mtp_update_acceleration_phase() {
//         let mut mtp = MTP::new(1.0, 10.0, 2.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         let v1 = mtp.update(Vector2::zeros(), Vector2::zeros(), DT);
//         assert_relative_eq!(v1.magnitude(), 1.0 * DT, epsilon = 1e-6);
//         let v2 = mtp.update(Vector2::zeros(), v1, DT);
//         assert_relative_eq!((v2 - v1).magnitude(), 1.0 * DT, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_mtp_update_cruise_phase() {
//         let mut mtp = MTP::new(1.0, 10.0, 2.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         let v = mtp.update(Vector2::zeros(), Vector2::new(10.0, 0.0), DT);
//         assert_relative_eq!(v.magnitude(), 10.0, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_mtp_update_deceleration_phase() {
//         let mut mtp = MTP::new(1.0, 10.0, 2.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         let v = mtp.update(Vector2::new(95.0, 0.0), Vector2::new(10.0, 0.0), DT);
//         assert!(
//             v.magnitude() < 10.0,
//             "Velocity should decrease ({} < 10.0)",
//             v.magnitude()
//         );
//         assert_relative_eq!(10.0 - v.magnitude(), 2.0 * DT, epsilon = 1e-6);
//     }

//     #[test]
//     fn test_mtp_update_multiple_steps() {
//         let mut mtp = MTP::new(1.0, 10.0, 2.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         let mut position = Vector2::zeros();
//         let mut velocity = Vector2::zeros();
//         for _ in 0..60 {
//             // Simulate 1 second
//             let new_velocity = mtp.update(position, velocity, DT);
//             assert!(
//                 (new_velocity - velocity).magnitude() <= 1.0 * DT + 1e-6,
//                 "Velocity change exceeds maximum acceleration"
//             );
//             position += (velocity + new_velocity) * 0.5 * DT; // Trapezoidal integration
//             velocity = new_velocity;
//         }
//         assert!(
//             position.magnitude() > 0.0 && position.magnitude() < 100.0,
//             "Position after 1 second should be between 0 and 100"
//         );
//         assert!(
//             velocity.magnitude() > 0.0 && velocity.magnitude() <= 10.0,
//             "Velocity after 1 second should be between 0 and 10"
//         );
//     }

//     #[test]
//     fn test_velocity_change_limit() {
//         let mut mtp = MTP::new(1.0, 10.0, 1.0);
//         mtp.set_setpoint(Vector2::new(100.0, 0.0));
//         for initial_velocity in [0.0, 5.0, 10.0] {
//             let initial_velocity_vec = Vector2::new(initial_velocity, 0.0);
//             let new_velocity = mtp.update(Vector2::zeros(), initial_velocity_vec, DT);
//             assert!(
//                 (new_velocity - initial_velocity_vec).magnitude() <= 1.0 * DT + 1e-6,
//                 "Velocity change exceeds maximum acceleration ({} -> {})",
//                 initial_velocity_vec.magnitude(),
//                 new_velocity.magnitude()
//             );
//         }
//     }
// }
