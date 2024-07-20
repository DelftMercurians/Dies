use dies_core::Angle;

pub struct YawController {
    kp: f64,
    cutoff_distance: f64,
    target_yaw: Option<Angle>,
}

impl YawController {
    pub fn new(kp: f64, cutoff_distance: f64) -> Self {
        YawController {
            cutoff_distance,
            kp,
            target_yaw: None,
        }
    }

    pub fn set_setpoint(&mut self, setpoint: Angle) {
        self.target_yaw = Some(setpoint);
    }

    pub fn update(
        &self,
        current_yaw: Angle,
        current_angular_velocity: f64,
        dt: f64,
        max_angular_velocity: f64,
        max_angular_acceleration: f64,
        carefullness: f64
    ) -> f64 {
        if let Some(target_yaw) = self.target_yaw {
            let error = target_yaw - current_yaw;

            if error.radians().abs() < self.cutoff_distance {
                return 0.0;
            }

            // Calculate the desired angular velocity
            let desired_angular_velocity = self.kp * error.radians();
            let acc = max_angular_acceleration * (1.0 - carefullness * 0.5);
            let vel = max_angular_velocity * (1.0 - carefullness * 0.8);

            // Limit the desired angular velocity and acceleration
            let acceleration = (desired_angular_velocity - current_angular_velocity) / dt;//.clamp(-acc, acc);

            (current_angular_velocity + acceleration * dt)
                .clamp(-vel, vel)
        } else {
            0.0
        }
    }

    pub fn update_settings(&mut self, kp: f64, cutoff_distance: f64) {
        self.kp = kp;
        self.cutoff_distance = cutoff_distance;
    }
}

// #[cfg(test)]
// mod tests {
//     use std::f64::consts::PI;

//     use super::*;

//     const MAX_VELOCITY: f64 = 1.0;
//     const MAX_ACCELERATION: f64 = 0.5;
//     const CUTOFF_DISTANCE: f64 = 0.01;
//     const DT: f64 = 1.0;

//     fn create_controller(target: f64) -> YawController {
//         let mut controller = YawController::new(MAX_VELOCITY, MAX_ACCELERATION, CUTOFF_DISTANCE);
//         controller.set_setpoint(Angle::from_radians(target));
//         controller
//     }

//     #[test]
//     fn test_within_cutoff_distance() {
//         let controller = create_controller(0.0);
//         let output = controller.update(Angle::from_radians(0.005), 0.0, DT);
//         assert_eq!(output, 0.0);
//     }

//     #[test]
//     fn test_acceleration() {
//         let controller = create_controller(PI / 2.0);
//         let output = controller.update(Angle::from_radians(0.0), 0.0, DT);
//         assert!(output > 0.0 && output <= MAX_ACCELERATION);
//     }

//     #[test]
//     fn test_deceleration() {
//         let controller = create_controller(0.0);
//         let output = controller.update(Angle::from_radians(0.1), 0.5, DT);
//         assert!((0.0..0.5).contains(&output));
//     }

//     #[test]
//     fn test_max_velocity() {
//         let controller = create_controller(PI);
//         let output = controller.update(Angle::from_radians(0.0), 0.9, DT);
//         assert_eq!(output, MAX_VELOCITY);
//     }

//     #[test]
//     fn test_wrap_around_positive() {
//         let controller = create_controller(PI);
//         let output = controller.update(Angle::from_radians(-3.0), 0.0, DT);
//         assert!(output < 0.0);
//     }

//     #[test]
//     fn test_wrap_around_negative() {
//         let controller = create_controller(-PI);
//         let output = controller.update(Angle::from_radians(3.0), 0.0, DT);
//         assert!(output > 0.0);
//     }

//     #[test]
//     fn test_near_negative_180_degrees() {
//         let controller = create_controller(PI / 2.0);
//         let output = controller.update(Angle::from_degrees(-179.0), 0.0, DT);
//         assert!(output < 0.0);
//     }

//     #[test]
//     fn test_shortest_path_clockwise() {
//         let controller = create_controller(-3.0 * PI / 4.0);
//         let output = controller.update(Angle::from_radians(3.0 * PI / 4.0), 0.0, DT);
//         assert!(output > 0.0);
//     }

//     #[test]
//     fn test_shortest_path_counterclockwise() {
//         let controller = create_controller(3.0 * PI / 4.0);
//         let output = controller.update(Angle::from_radians(-3.0 * PI / 4.0), 0.0, DT);
//         assert!(output < 0.0);
//     }

//     #[test]
//     fn test_overshoot() {
//         let controller = create_controller(PI / 2.0);
//         let output = controller.update(Angle::from_radians((PI / 2.0) + 0.05), 0.1, DT);
//         assert!(output < 0.0);
//     }

//     #[test]
//     fn test_update_settings() {
//         let mut controller = create_controller(0.0);
//         controller.update_settings(2.0, 1.0, 0.02);
//         let output = controller.update(Angle::from_radians(PI / 2.0), 0.0, DT);
//         assert_eq!(output, -1.0);
//     }

//     #[test]
//     fn test_set_target_yaw() {
//         let mut controller = create_controller(0.0);
//         controller.set_setpoint(Angle::from_radians(PI));
//         let output = controller.update(Angle::from_radians(0.0), 0.0, DT);
//         assert!(output > 0.0);
//     }
// }
