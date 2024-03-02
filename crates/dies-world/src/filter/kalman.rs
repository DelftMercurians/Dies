use nalgebra as na;
use na::{DimName, OMatrix, U2,};
use na::{Const, DefaultAllocator, SMatrix};
use na::allocator::Allocator;
use na::OVector;
use crate::filter::matrix_gen::MatrixCreator;

// OS: Observation Space, SS: State Space
pub struct Kalman<OS,SS>
where
    OS: DimName,
    SS: DimName,
    DefaultAllocator: Allocator<f64, SS, SS> + Allocator<f64, OS, SS>
    + Allocator<f64, SS, OS> + Allocator<f64, OS, OS>
    + Allocator<f64, SS> + Allocator<f64, OS>,
{
    // unit standard deviation of transition noise
    std: f64,
    // the time for the current state in ms
    t: f64,
    // Transition matrix
    A: Box<dyn MatrixCreator<SS>>,
    // Transformation (observation) matrix
    H: OMatrix<f64, OS, SS>,
    // Process noise covariance matrix
    Q: Box<dyn MatrixCreator<SS>>,
    // Measurement noise covariance matrix
    R: OMatrix<f64, OS, OS>,
    // Error covariance matrix
    P: OMatrix<f64, SS, SS>,
    // State vector
    x: OVector<f64, SS>,
}

impl <OS,SS> Kalman<OS,SS>
where
    OS: DimName,
    SS: DimName,
    DefaultAllocator: Allocator<f64, SS, SS> + Allocator<f64, OS, SS>
    + Allocator<f64, SS, OS> + Allocator<f64, OS, OS>
    + Allocator<f64, SS> + Allocator<f64, OS>,
{


    pub fn new(
        std: f64,
        t: f64,
        A: Box<dyn MatrixCreator<SS>>,
        H: OMatrix<f64, OS, SS>,
        Q: Box<dyn MatrixCreator<SS>>,
        R: OMatrix<f64, OS, OS>,
        P: OMatrix<f64, SS, SS>,
        x: OVector<f64, SS>,
    ) -> Self {
        Kalman {
            std,
            t,
            A,
            H,
            Q,
            R,
            P,
            x,
        }
    }

    //predict a future state(prior), this doesn't change the internal state of the filter
    pub fn predict(&mut self, newt: f64) -> OVector<f64, SS> {
        &self.A.create_matrix(newt - self.t) * &self.x
    }

    //update the state of the filter based on a new observation and a new time
    //returns None if the packet is dropped, otherwise return the posterior state
    pub fn update(&mut self, z: OVector<f64, OS>, newt: f64) -> Option<OVector<f64, SS>>{
        let dt = newt - self.t;
        if dt < 0.0 {
            return None;
        }
        let A = self.A.create_matrix(dt);
        let Q = self.Q.create_matrix(dt);
        let x = &A * &self.x;
        let P = &A * &self.P * &A.transpose() + &Q*self.std.powi(2);
        let r = z - &self.H * &x;
        let S = &self.H * &P * &self.H.transpose() + &self.R;
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
    use nalgebra::U4;
    use super::*;

    use crate::filter::filter_builder::KalmanBuilder;

    #[test]
    fn test_kalman() {
        let dt = 1.0/40.0; //40fps
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
        let mut true_position =  Vec::new();
        true_position.push(OVector::<f64, U2>::new(0.0, 0.0));
        for i in 1..num_steps {
            let x = velocity.0 * dt + measurements[i-1][0];
            let y = velocity.1 * dt + measurements[i-1][1];
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