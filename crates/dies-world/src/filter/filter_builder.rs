use crate::filter::kalman::Kalman;
use crate::filter::matrix_gen::{ULMotionModel, WhiteNoise1stOrder};
use na::OVector;
use na::U4;
use na::{OMatrix, U2};
use nalgebra as na;

pub struct KalmanBuilder {
    init_std: f64,
    measurement_std: f64,
    unit_transition_std: f64,
}

impl KalmanBuilder {
    pub fn new(init_std: f64, measurement_std: f64, unit_transition_std: f64) -> Self {
        KalmanBuilder {
            init_std,
            measurement_std,
            unit_transition_std,
        }
    }
    #[allow(non_snake_case)]
    pub fn build2D(&self, init_pos: OVector<f64, U4>, int_time: f64) -> Kalman<U2, U4> {
        let std = self.unit_transition_std;
        let t = int_time;
        #[allow(non_snake_case)]
        let A = ULMotionModel;
        let H: OMatrix<f64, U2, U4> =
            OMatrix::<f64, U2, U4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let Q = WhiteNoise1stOrder;
        #[allow(non_snake_case)]
        let R: OMatrix<f64, U2, U2> = OMatrix::<f64, U2, U2>::new(
            self.measurement_std.powi(2),
            0.0,
            0.0,
            self.measurement_std.powi(2),
        );
        #[allow(non_snake_case)]
        let P: OMatrix<f64, U4, U4> = OMatrix::<f64, U4, U4>::identity() * self.init_std.powi(2);
        let x: OVector<f64, U4> = init_pos;

        Kalman::new(std, t, Box::new(A), H, Box::new(Q), R, P, x)
    }
}
