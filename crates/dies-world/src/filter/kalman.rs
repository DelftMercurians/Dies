use crate::filter::matrix_gen::MatrixCreator;
use na::allocator::Allocator;
use na::DefaultAllocator;
use na::OVector;
use na::{DimName, OMatrix, U1};
use nalgebra as na;

// OS: Observation Space, SS: State Space
pub struct Kalman<OS, SS>
where
    OS: DimName,
    SS: DimName,
    DefaultAllocator: Allocator<f64, SS, SS>
        + Allocator<f64, OS, SS>
        + Allocator<f64, SS, OS>
        + Allocator<f64, OS, OS>
        + Allocator<f64, SS>
        + Allocator<f64, OS>,
{
    // variance of transition noise
    var: f64,
    // the time for the current state in ms
    t: f64,
    // Transition matrix
    A: Box<dyn MatrixCreator<SS,SS>>,
    // Transformation (observation) matrix
    H: OMatrix<f64, OS, SS>,
    // Process noise covariance matrix
    Q: Box<dyn MatrixCreator<SS, SS>>,
    // Measurement noise covariance matrix
    R: OMatrix<f64, OS, OS>,
    // Error covariance matrix
    P: OMatrix<f64, SS, SS>,
    // State vector
    x: OVector<f64, SS>,
    // Control input
    B: Option<Box<dyn MatrixCreator<SS, U1>>>
}

impl<OS, SS> Kalman<OS, SS>
where
    OS: DimName,
    SS: DimName,
    DefaultAllocator: Allocator<f64, SS, SS>
        + Allocator<f64, OS, SS>
        + Allocator<f64, SS, OS>
        + Allocator<f64, OS, OS>
        + Allocator<f64, SS>
        + Allocator<f64, OS>,
{
    pub fn new(
        var: f64,
        t: f64,
        A: Box<dyn MatrixCreator<SS, SS>>,
        H: OMatrix<f64, OS, SS>,
        Q: Box<dyn MatrixCreator<SS,SS>>,
        R: OMatrix<f64, OS, OS>,
        P: OMatrix<f64, SS, SS>,
        x: OVector<f64, SS>,
        B: Option<Box<dyn MatrixCreator<SS, U1>>>,
    ) -> Self {
        Kalman {
            var,
            t,
            A,
            H,
            Q,
            R,
            P,
            x,
            B,
        }
    }

    //predict a future state(prior), this doesn't change the internal state of the filter
    #[allow(dead_code)]
    pub fn predict(&mut self, newt: f64) -> OVector<f64, SS> {
        let r = &self.A.create_matrix(newt - self.t) * &self.x;
        if let Some(B) = &self.B {
            r + B.create_matrix(newt - self.t)
        } else {
            r
        }
    }

    //update the state of the filter based on a new observation and a new time
    //returns None if the packet is dropped, otherwise return the posterior state
    pub fn update(&mut self, z: OVector<f64, OS>, newt: f64) -> Option<OVector<f64, SS>> {
        let dt = newt - self.t;
        if dt < 0.0 {
            return None;
        }
        #[allow(non_snake_case)]
        let A = self.A.create_matrix(dt);
        #[allow(non_snake_case)]
        let Q = self.Q.create_matrix(dt);
        let mut x = &A * &self.x;
        if let Some(B) = &self.B {
            x += B.create_matrix(dt);
        }
        let P = &A * &self.P * &A.transpose() + &Q * self.var;
        let r = z - &self.H * &x;
        #[allow(non_snake_case)]
        let S = &self.H * &P * &self.H.transpose() + &self.R;
        #[allow(non_snake_case)]
        let K = &P * &self.H.transpose() * S.try_inverse().unwrap();
        self.x = x + &K * r;
        self.P = &P - &K * &self.H * &P;
        self.t = newt;
        Some(self.x.clone())
    }
}

//test
#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{U2, U4};

    use crate::filter::filter_builder::KalmanBuilder;

    #[test]
    fn test_kalman() {
        let dt = 1.0 / 40.0; //40fps
        let init_pos = OVector::<f64, U4>::new(0.0, 0.0, 0.0, 0.0);
        let init_time = 0.0;
        let init_std = 0.1;
        let measurement_std = 0.1;
        let unit_transition_std = 2.0;
        let builder = KalmanBuilder::new(init_std, measurement_std, unit_transition_std);
        let mut filter = builder.build2D(init_pos, init_time);

        // generate some fake measurements trajectory
        // with constant speed movement
        let simulation_time = 0.5;
        let num_steps = (simulation_time / dt) as usize;
        let velocity = (0.5, 0.5);
        let mut measurements = Vec::new();
        measurements.push(OVector::<f64, U2>::new(0.0, 0.0));
        let mut true_position = Vec::new();
        true_position.push(OVector::<f64, U2>::new(0.0, 0.0));
        for i in 1..num_steps {
            let x = velocity.0 * dt + measurements[i - 1][0];
            let y = velocity.1 * dt + measurements[i - 1][1];
            let x_measured = x + measurement_std * rand::random::<f64>();
            let y_measured = y + measurement_std * rand::random::<f64>();
            measurements.push(OVector::<f64, U2>::new(x_measured, y_measured));
            true_position.push(OVector::<f64, U2>::new(x, y));
        }
        let mut filtered = Vec::new();
        filtered.push(OVector::<f64, U2>::new(0.0, 0.0));
        for i in 1..num_steps {
            let newt = i as f64 * dt;
            let z = measurements[i];
            let state = filter.update(z, newt);
            if let Some(s) = state {
                filtered.push(OVector::<f64, U2>::new(s[0], s[2]));
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
}
