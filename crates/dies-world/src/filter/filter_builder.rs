use crate::filter::kalman::Kalman;
use crate::filter::matrix_gen::{GravityControl, Piecewise1stOrder, ULMotionModel, WhiteNoise1stOrder};
use nalgebra::{SMatrix, SVector};
#[derive(Debug)]
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
    pub fn build2D(&self, init_pos: SVector<f64, 4>, int_time: f64) -> Kalman<2, 4> {
        let var = self.unit_transition_var;
        let t = int_time;
        #[allow(non_snake_case)]
        let A = ULMotionModel;
        let H: SMatrix<f64, 2, 4> =
            SMatrix::<f64, 2, 4>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let Q = WhiteNoise1stOrder;
        #[allow(non_snake_case)]
        let R: SMatrix<f64, 2, 2> = SMatrix::<f64, 2, 2>::new(
            self.measurement_var,
            0.0,
            0.0,
            self.measurement_var,
        );
        #[allow(non_snake_case)]
        let P: SMatrix<f64, 4, 4> = SMatrix::<f64, 4, 4>::identity() * self.init_var;
        let x: SVector<f64, 4> = init_pos;

        Kalman::new(var, t, Box::new(A), H, Box::new(Q), R, P, x, None)
    }

    pub fn build_3d(&self, init_state: SVector<f64, 6>, int_time: f64) -> Kalman<3, 6> {
        let var = self.unit_transition_var;
        let t = int_time;
        #[allow(non_snake_case)]
        let A = ULMotionModel;
        let H: SMatrix<f64, 3, 6> =
            SMatrix::<f64, 3, 6>::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

        let Q = Piecewise1stOrder;
        #[allow(non_snake_case)]
            let R: SMatrix<f64, 3, 3> = SMatrix::<f64, 3, 3>::new(
            self.measurement_var,
            0.0,
            0.0,
            0.0,
            self.measurement_var,
            0.0,
            0.0,
            0.0,
            self.measurement_var,
        );
        #[allow(non_snake_case)]
        let P: SMatrix<f64, 6, 6> = SMatrix::<f64, 6, 6>::identity() * self.init_var;
        let x: SVector<f64, 6> = init_state;
        let B = GravityControl;

        Kalman::new(var, t, Box::new(A), H, Box::new(Q), R, P, x, Option::Some(Box::new(B)))
    }
}
