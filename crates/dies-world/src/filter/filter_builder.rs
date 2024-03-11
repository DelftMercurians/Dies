use crate::filter::kalman::Kalman;
use crate::filter::matrix_gen::{ULMotionModel, WhiteNoise1stOrder};
use na::OVector;
use na::U4;
use na::{OMatrix, U2};
use nalgebra as na;

pub struct KalmanBuilder {
    init_var: f64,
    measurement_var: f64,
    unit_transition_var: f64,
}

impl KalmanBuilder {
    pub fn new(init_var: f64, measurement_var: f64, unit_transition_var: f64) -> Self {
        KalmanBuilder {
            init_var,
            measurement_var,
            unit_transition_var,
        }
    }
    #[allow(non_snake_case)]
    pub fn build2D(&self, init_pos: OVector<f64, U4>, int_time: f64) -> Kalman<U2, U4> {
        let var = self.unit_transition_var;
        let t = int_time;
        #[allow(non_snake_case)]
        let A = ULMotionModel;
        let H: OMatrix<f64, U2, U4> =
            OMatrix::<f64, U2, U4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let Q = WhiteNoise1stOrder;
        #[allow(non_snake_case)]
        let R: OMatrix<f64, U2, U2> = OMatrix::<f64, U2, U2>::new(
            self.measurement_var,
            0.0,
            0.0,
            self.measurement_var,
        );
        #[allow(non_snake_case)]
        let P: OMatrix<f64, U4, U4> = OMatrix::<f64, U4, U4>::identity() * self.init_var;
        let x: OVector<f64, U4> = init_pos;

        Kalman::new(var, t, Box::new(A), H, Box::new(Q), R, P, x)
    }
}
