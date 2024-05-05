use std::time::Instant;

use super::variable::Variable;

pub struct PID<T> {
    kp: f32,
    ki: f32,
    kd: f32,
    setpoint: Option<T>,
    integral: T,
    last_error: Option<T>,
    last_time: Instant,
}

impl<T> PID<T>
where
    T: Variable,
{
    pub fn new(kp: f32, ki: f32, kd: f32) -> Self {
        Self {
            kp,
            ki,
            kd,
            setpoint: None,
            integral: T::zero(),
            last_error: None,
            last_time: Instant::now(),
        }
    }

    pub fn set_setpoint(&mut self, setpoint: T) {
        self.setpoint = Some(setpoint);
    }

    pub fn update(&mut self, input: T) -> T {
        if let Some(setpoint) = self.setpoint {
            let error = setpoint - input;
            let dt = self.last_time.elapsed().as_secs_f32();
            self.last_time = Instant::now();

            self.integral += error * dt;

            let last_error = self.last_error.unwrap_or(error);
            let derivative = (error - last_error) / dt;
            self.last_error = Some(error);

            error * self.kp + self.integral * self.ki + derivative * self.kd
        } else {
            T::zero()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector2;

    #[test]
    fn test_pid_update_without_setpoint() {
        let mut pid = PID::<f32>::new(1.0, 1.0, 1.0);

        let output = pid.update(1.0);

        assert_eq!(output, 0.0);
    }

    #[test]
    fn test_pid_update_with_setpoint() {
        let mut pid = PID::<f32>::new(1.0, 1.0, 1.0);
        pid.set_setpoint(2.0);

        let output = pid.update(1.0);
        assert_ne!(output, 0.0);
    }

    #[test]
    fn test_update_with_vector() {
        let mut pid = PID::new(1.0, 1.0, 1.0);
        pid.set_setpoint(Vector2::new(2.0, 2.0));

        let output = pid.update(Vector2::new(1.0, 1.0));
        assert_ne!(output, Vector2::zeros());
    }
}
