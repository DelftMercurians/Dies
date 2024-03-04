use nalgebra as na;
use na::{DimName, OMatrix, U2, U4};
use na::{DefaultAllocator};
use na::allocator::Allocator;

fn block_diag(
    a: &OMatrix<f64, U2, U2>,
) -> OMatrix<f64, U4, U4>
{
    let mut result = OMatrix::<f64, U4, U4>::zeros();

    result.fixed_view_mut::<2, 2>(0, 0).copy_from(a);
    result.fixed_view_mut::<2, 2>(2, 2).copy_from(a);
    result
}

pub trait MatrixCreator<D>
    where
        D: DimName,
        DefaultAllocator: Allocator<f64, D, D>,
{
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64, D, D>;
}

pub struct Piecewise1stOrder;

// 2x2 measurement noise matrix assuming acceleration is different in
// different duration
impl MatrixCreator<U2> for Piecewise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64, U2, U2> {
        OMatrix::<f64, U2, U2>::new(
            delta_t.powi(4)/4.0, delta_t.powi(3)/2.0,
            delta_t.powi(3)/2.0, delta_t.powi(2),
        )
    }
}

//Piecewise1stOrder but 2D
impl MatrixCreator<U4> for Piecewise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64,U4,U4> {
        let m = OMatrix::<f64, U2, U2>::new(
            delta_t.powi(4)/4.0, delta_t.powi(3)/2.0,
            delta_t.powi(3)/2.0, delta_t.powi(2),
        );
        block_diag(&m)
    }
}

pub struct WhiteNoise1stOrder;

// 2x2 measurement noise matrix assuming acceleration does not change
impl MatrixCreator<U2> for WhiteNoise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64, U2, U2> {
        OMatrix::<f64, U2, U2>::new(
            delta_t.powi(3)/3.0,  delta_t.powi(2)/2.0,
            delta_t.powi(2)/2.0, delta_t,
        )
    }
}

//WhiteNoise1stOrder but 2D
impl MatrixCreator<U4> for WhiteNoise1stOrder {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64,U4,U4> {
        let m = OMatrix::<f64, U2, U2>::new(
            delta_t.powi(3)/3.0,  delta_t.powi(2)/2.0,
            delta_t.powi(2)/2.0, delta_t,
        );
        block_diag(&m)
    }
}

pub struct ULMotionModel;

// 2x2 transition matrix assuming constant speed
impl MatrixCreator<U2> for ULMotionModel {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64, U2, U2> {
        OMatrix::<f64, U2, U2>::new(
            1.0, delta_t,
            0.0, 1.0,
        )
    }
}

//ULMotionModel but 2D
impl MatrixCreator<U4> for ULMotionModel {
    fn create_matrix(&self, delta_t: f64) -> OMatrix<f64,U4,U4> {
        let m = OMatrix::<f64, U2, U2>::new(
            1.0, delta_t,
            0.0, 1.0,
        );
        block_diag(&m)
    }
}




#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_white_noise_1st_order_u2() {
        let wn = WhiteNoise1stOrder;
        let delta_t = 2.0_f64;
        let result_matrix:OMatrix<f64,U2,U2> = wn.create_matrix(delta_t);
        let expected_matrix = OMatrix::<f64, U2, U2>::new(
            8.0/3.0, 4.0/2.0,
            4.0/2.0, 2.0,
        );

        assert!((result_matrix - expected_matrix).norm() < 1e-6);
    }

    #[test]
    fn test_white_noise_1st_order_u4() {
        let wn = WhiteNoise1stOrder;
        let delta_t = 1.0_f64;
        let result_matrix:OMatrix<f64,U4,U4> = wn.create_matrix(delta_t);
        let expected_matrix = OMatrix::<f64, U4, U4>::new(
            1.0/3.0, 1.0/2.0, 0.0, 0.0,
            1.0/2.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0/3.0, 1.0/2.0,
            0.0, 0.0, 1.0/2.0, 1.0,
        );

        assert_eq!(result_matrix, expected_matrix);
    }
}

