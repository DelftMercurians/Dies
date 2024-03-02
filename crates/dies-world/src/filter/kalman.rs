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
{
    // unit standard deviation of transition noise
    std: f64,
    // the time for the current state in ms
    t: f64,
    // Transition matrix
    A: dyn MatrixCreator<SS>,
    // Transformation (observation) matrix
    H: OMatrix<f64, OS, SS>,
    // Process noise covariance matrix
    Q: dyn MatrixCreator<SS>,
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
    DefaultAllocator: Allocator<f64, OS>,
    DefaultAllocator: Allocator<f64, SS>,
    DefaultAllocator: Allocator<f64, SS, SS>,
    DefaultAllocator: Allocator<f64, OS, SS>,
    DefaultAllocator: Allocator<f64, SS, OS>,
    DefaultAllocator: Allocator<f64, SS>,
    DefaultAllocator: Allocator<f64, OS, OS>,
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

    //predict a future state, this doesn't change the internal state of the filter
    pub fn predict(&mut self, newt: f64) -> OVector<f64, SS> {
        &self.A.create_matrix(newt - self.t) * &self.x
    }

    //update the state of the filter based on a new observation and a new time
    //returns None if the packet is dropped
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