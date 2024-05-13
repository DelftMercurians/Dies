use nalgebra::{DMatrix, Dvector};
use optimization_engine::{alm::*, constraints::*, core::*, panoc::*};
// define the system's state and control input dimensions
const STATE_DIM: usize = 2;
const CONTROL_DIM: usize = 2;

struct MPCParams {
    q: Vec<f64>, //State cost weight
    r: Vec<f64>, // Control input cost weight
    dt: f64,
    n: usize, // add other params here
}
pub struct MPCController {
    params: MPCParams,
    optimizer: PANOCOptimizer,
}

impl MPCController {
    pub fn new(params: MPCParams) -> Self {
        let bounds = NoConstraints::new();
        let problem = Problem::new(
            &bounds,
            move |u, grad| cost_gradient(u, &params, grad),
            move |u| cost_function(u, &params),
        );
        let optimizer = PANOCOptimizer::new(problem, SolverOptions::default());
        Self { params, optimizer }
    }

    fn cost_function(u: &[f64], params: &MPCParams) -> f64 {
        // the cost function must be implemented here based on params and u.
    }
    fn cost_gradient(u: &[f64], params: &MPCParams, grad: &mut [f64]) {
        // calculate grad based on params and u.
    }

    pub fn solve(&mut self, initial_guess: &[f64]) -> Result<Vec<f64>, SolverError> {
        let mut u = initial_guess.to_vec(); // Copy initial guess to the optimization variable
        let res = self.optimizer.solve(&mut u)?;
        Ok(u) // Return the optimized control input
    }
}
