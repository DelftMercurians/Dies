use crate::filter::matrix_gen::{GravityControl, MatrixCreator, Piecewise1stOrder, ULMotionModel, WhiteNoise1stOrder};
use na::{SMatrix, SVector};
use nalgebra as na;

// OS: Observation Space, SS: State Space
#[derive(Debug)]
pub struct Kalman<const OS: usize, const SS: usize>
{
    var: f64,
    t: f64,
    transition_matrix: Box<dyn MatrixCreator<SS,SS>>,
    transformation_matrix: SMatrix<f64, OS, SS>,
    process_noise: Box<dyn MatrixCreator<SS, SS>>,
    measurement_noise: SMatrix<f64, OS, OS>,
    posteriori_covariance: SMatrix<f64, SS, SS>,
    x: SVector<f64, SS>,
    control: Option<Box<dyn MatrixCreator<SS, 1>>>
}

impl<const OS: usize,const SS: usize> Kalman<OS,SS>
{
    pub fn set_x(&mut self, x: SVector<f64, SS>) {
        self.x = x;
    }

    pub fn gating(&self, r: SVector<f64, OS>) -> bool {
        const GATE_LIMIT: f64 = 16.0;
        for i in 0..r.len() {
            if r[i] * r[i] > GATE_LIMIT * self.posteriori_covariance[(i, i)]{
                return false;
            }
        }
        return true;
    }

    pub fn update(&mut self, z: SVector<f64, OS>, newt: f64, use_gate: bool) -> Option<SVector<f64, SS>> {
        let dt = newt - self.t;
        if dt < 0.0 {
            return None;
        }
        let transition_matrix = self.transition_matrix.create_matrix(dt);
        let process_noise = self.process_noise.create_matrix(dt);
        let mut x = &transition_matrix * &self.x;
        if let Some(control) = &self.control {
            x += control.create_matrix(dt);
        }
        let r = z - &self.transformation_matrix * &x;
        if use_gate && !self.gating(r.clone_owned()) {
            return Option::from(x.clone());
        }

        let posteriori_covariance = &transition_matrix * &self.posteriori_covariance
            * &transition_matrix.transpose() + &process_noise * self.var;
        let innovation_covariance = &self.transformation_matrix * &posteriori_covariance
            * &self.transformation_matrix.transpose() + &self.measurement_noise;
        let kalman_gain = &posteriori_covariance * &self.transformation_matrix.transpose()
            * innovation_covariance.try_inverse().unwrap();
        self.x = x + &kalman_gain * r;
        self.posteriori_covariance = &posteriori_covariance - &kalman_gain * &self.transformation_matrix
            * &posteriori_covariance;
        self.t = newt;
        Some(self.x.clone())
    }
}

impl Kalman<2, 4> {
    pub fn new_player_filter(
        init_var: f64,
        unit_transition_var: f64,
        measurement_var: f64,
        init_pos: SVector<f64, 4>,
        int_time: f64
    ) -> Self {
        Kalman {
            var: unit_transition_var,
            t: int_time,
            transition_matrix: Box::new(ULMotionModel),
            transformation_matrix: SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 1.0, 0.0),
            process_noise: Box::new(WhiteNoise1stOrder),
            measurement_noise: SMatrix::<f64, 2, 2>::new(
                measurement_var,
                0.0,
                0.0,
                measurement_var,
            ),
            posteriori_covariance: SMatrix::<f64, 4, 4>::identity() * init_var,
            x: init_pos,
            control: None,
        }

    }
}

impl Kalman<3, 6> {
    pub fn new_ball_filter(
        init_var: f64,
        unit_transition_var: f64,
        measurement_var: f64,
        init_pos: SVector<f64, 6>,
        int_time: f64
    ) -> Self {
        Kalman {
            var: unit_transition_var,
            t: int_time,
            transition_matrix: Box::new(ULMotionModel),
            transformation_matrix: SMatrix::<f64, 3, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                                             0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            process_noise: Box::new(Piecewise1stOrder),
            measurement_noise: SMatrix::<f64, 3, 3>::new(
                measurement_var,
                0.0,
                0.0,
                0.0,
                measurement_var,
                0.0,
                0.0,
                0.0,
                measurement_var,
            ),
            posteriori_covariance: SMatrix::<f64, 6, 6>::identity() * init_var,
            x: init_pos,
            control: Some(Box::new(GravityControl))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_kalman() {
        let dt = 1.0/40.0; //40fps
        let init_pos = SVector::<f64, 4>::new(0.0, 0.0, 0.0, 0.0);
        let init_time:f64 = 0.0;
        let init_std:f64 = 100.0;
        let measurement_std:f64 = 5.0;
        let unit_transition_std:f64 = 200.0;
        let mut filter = Kalman::<2, 4>::new_player_filter(init_std.powi(2),
                                                           unit_transition_std.powi(2),
                                                           measurement_std.powi(2), init_pos, init_time);

        // generate some fake measurements trajectory
        // with constant speed movement
        let simulation_time:f64 = 10.0;
        let num_steps = (simulation_time / dt) as usize;
        let velocity = (500.0, 500.0);
        let mut measurements = Vec::new();
        measurements.push(SVector::<f64, 2>::new(0.0, 0.0));
        let mut true_position = Vec::new();
        true_position.push(SVector::<f64, 2>::new(0.0, 0.0));
        for i in 1..num_steps {
            let x = velocity.0 * dt + true_position[i - 1][0];
            let y = velocity.1 * dt + true_position[i - 1][1];
            let x_measured = x + measurement_std * rand::random::<f64>();
            let y_measured = y + measurement_std * rand::random::<f64>();
            measurements.push(SVector::<f64, 2>::new(x_measured, y_measured));
            true_position.push(SVector::<f64, 2>::new(x, y));
        }
        let mut filtered = Vec::new();
        filtered.push(SVector::<f64, 2>::new(0.0, 0.0));
        for i in 1..num_steps {
            let newt = i as f64 * dt;
            let z = measurements[i];
            let state = filter.update(z, newt, false);
            if let Some(s) = state {
                filtered.push(SVector::<f64, 2>::new(s[0], s[2]));
            }
        }

        //calulate the error
        let mut smoothed_error = 0.0;
        let mut original_error = 0.0;
        for i in 1..num_steps {
            let e = (filtered[i] - true_position[i]).norm();
            let e2 = (measurements[i] - true_position[i]).norm();
            smoothed_error += e;
            original_error += e2;
        }
        // println!("Smoothed error: {:.2}, Original error: {:.2}", smoothed_error, original_error);
        assert!(smoothed_error < original_error);
    }

    #[test]
    fn test_kalman_3d() {
        let dt:f64 = 1.0 / 40.0;
        let init_pos = SVector::<f64, 6>::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let init_time: f64 = 0.0;
        let init_std: f64 = 50.0;
        let measurement_std: f64 = 5.0;
        let unit_transition_std: f64 = 200.0;
        let mut filter = Kalman::<3, 6>::new_ball_filter(init_std.powi(2),
                                                         unit_transition_std.powi(2),
                                                         measurement_std.powi(2),
                                                         init_pos, init_time);
        // generate some fake measurements trajectory
        let simulation_time = 1.5;
        let num_steps = (simulation_time / dt) as usize;
        let mut velocity = (1000.0, 1000.0, 5000.0);
        let mut measurements = Vec::new();
        measurements.push(SVector::<f64, 3>::new(0.0, 0.0, 0.0));
        let mut true_position = Vec::new();
        true_position.push(SVector::<f64, 3>::new(0.0, 0.0, 0.0));
        for i in 1..num_steps {
            let x = velocity.0 * dt + true_position[i - 1][0];
            let y = velocity.1 * dt + true_position[i - 1][1];
            let z = velocity.2 * dt + true_position[i - 1][2] - 9810.0 * dt.powi(2) / 2.0;
            let x_measured = x + measurement_std * rand::random::<f64>();
            let y_measured = y + measurement_std * rand::random::<f64>();
            let z_measured = z + measurement_std * rand::random::<f64>();
            measurements.push(SVector::<f64, 3>::new(x_measured, y_measured, z_measured));
            true_position.push(SVector::<f64, 3>::new(x, y, z));
        }
        let mut filtered:Vec<SVector<f64, 3>> = Vec::new();
        filtered.push(SVector::<f64, 3>::new(0.0, 0.0, 0.0));
        for i in 1..num_steps {
            let newt = i as f64 * dt;
            let z = measurements[i];
            let state = filter.update(z, newt, false);
            if let Some(s) = state {
                filtered.push(SVector::<f64, 3>::new(s[0], s[2], s[4]));
            }
        }

        //calulate the error
        let mut smoothed_error = 0.0;
        let mut original_error = 0.0;
        for i in 1..num_steps {
            let e = (filtered[i] - true_position[i]).norm();
            let e2 = (measurements[i] - true_position[i]).norm();
            smoothed_error += e;
            original_error += e2;
        }
        assert!(smoothed_error < original_error);
    }

}
