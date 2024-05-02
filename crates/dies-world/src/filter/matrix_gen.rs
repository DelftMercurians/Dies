use na::SMatrix;
use nalgebra as na;
use std::fmt::Debug;

/// Create a block diagonal matrix from two 2x2 matrix
fn block_diag(a: &SMatrix<f64, 2, 2>) -> SMatrix<f64, 4, 4> {
    let mut result = SMatrix::<f64, 4, 4>::zeros();
    result.fixed_view_mut::<2, 2>(0, 0).copy_from(a);
    result.fixed_view_mut::<2, 2>(2, 2).copy_from(a);
    result
}

/// Create a block diagonal matrix from three 2x2 matrix
fn block_diag_3(a: &SMatrix<f64, 2, 2>) -> SMatrix<f64, 6, 6> {
    let mut result = SMatrix::<f64, 6, 6>::zeros();
    result.fixed_view_mut::<2, 2>(0, 0).copy_from(a);
    result.fixed_view_mut::<2, 2>(2, 2).copy_from(a);
    result.fixed_view_mut::<2, 2>(4, 4).copy_from(a);
    result
}

pub trait MatrixCreator<const D1: usize, const D2: usize>: Debug + Send {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, D1, D2>;

    fn print(&self) {
        println!("{:?}", self.create_matrix(1.0));
    }
}

#[derive(Debug)]
pub struct Piecewise1stOrder;

/// 2x2 measurement noise matrix assuming acceleration is different in
/// different duration
impl MatrixCreator<2, 2> for Piecewise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::new(
            delta_t.powi(4) / 4.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(2),
        )
    }
}

///Piecewise1stOrder but 2D
impl MatrixCreator<4, 4> for Piecewise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 4, 4> {
        let m = SMatrix::<f64, 2, 2>::new(
            delta_t.powi(4) / 4.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(2),
        );
        block_diag(&m)
    }
}

impl MatrixCreator<6, 6> for Piecewise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 6, 6> {
        let m = SMatrix::<f64, 2, 2>::new(
            delta_t.powi(4) / 4.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(3) / 2.0,
            delta_t.powi(2),
        );
        block_diag_3(&m)
    }
}

#[derive(Debug)]
pub struct WhiteNoise1stOrder;

/// 2x2 measurement noise matrix assuming acceleration does not change
impl MatrixCreator<2, 2> for WhiteNoise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::new(
            delta_t.powi(3) / 3.0,
            delta_t.powi(2) / 2.0,
            delta_t.powi(2) / 2.0,
            delta_t,
        )
    }
}

///WhiteNoise1stOrder but 2D
impl MatrixCreator<4, 4> for WhiteNoise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 4, 4> {
        let m = SMatrix::<f64, 2, 2>::new(
            delta_t.powi(3) / 3.0,
            delta_t.powi(2) / 2.0,
            delta_t.powi(2) / 2.0,
            delta_t,
        );
        block_diag(&m)
    }
}

#[derive(Debug)]
pub struct ULMotionModel;

// 2x2 transition matrix assuming constant speed
impl MatrixCreator<2, 2> for ULMotionModel {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 2, 2> {
        SMatrix::<f64, 2, 2>::new(1.0, delta_t, 0.0, 1.0)
    }
}

///ULMotionModel but 2D
impl MatrixCreator<4, 4> for ULMotionModel {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 4, 4> {
        let m = SMatrix::<f64, 2, 2>::new(1.0, delta_t, 0.0, 1.0);
        block_diag(&m)
    }
}

/// ULMotionModel in xy, parabola z
impl MatrixCreator<6, 6> for ULMotionModel {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 6, 6> {
        let m = SMatrix::<f64, 2, 2>::new(1.0, delta_t, 0.0, 1.0);
        block_diag_3(&m)
    }
}

#[derive(Debug)]
pub struct GravityControl;

/// control input matrix for vel z adjustment
impl MatrixCreator<6, 1> for GravityControl {
    fn create_matrix(&self, delta_t: f64) -> SMatrix<f64, 6, 1> {
        //in mm/s^2
        let g: f64 = 9810.0;
        SMatrix::<f64, 6, 1>::new(0.0, 0.0, 0.0, 0.0, -g * delta_t.powi(2) / 2.0, -g * delta_t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_white_noise_1st_order_u2() {
        let wn = WhiteNoise1stOrder;
        let delta_t = 2.0_f64;
        let result_matrix: SMatrix<f64, 2, 2> = wn.create_matrix(delta_t);
        let expected_matrix = SMatrix::<f64, 2, 2>::new(8.0 / 3.0, 4.0 / 2.0, 4.0 / 2.0, 2.0);

        assert!((result_matrix - expected_matrix).norm() < 1e-6);
    }

    #[test]
    fn test_white_noise_1st_order_u4() {
        let wn = WhiteNoise1stOrder;
        let delta_t = 1.0_f64;
        let result_matrix: SMatrix<f64, 4, 4> = wn.create_matrix(delta_t);
        let expected_matrix = SMatrix::<f64, 4, 4>::new(
            1.0 / 3.0,
            1.0 / 2.0,
            0.0,
            0.0,
            1.0 / 2.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 3.0,
            1.0 / 2.0,
            0.0,
            0.0,
            1.0 / 2.0,
            1.0,
        );

        assert_eq!(result_matrix, expected_matrix);
    }
}
