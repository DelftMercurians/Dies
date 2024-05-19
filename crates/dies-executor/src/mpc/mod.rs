use dies_core::WorldData;
use optimization_engine::{
    constraints::NoConstraints,
    panoc::{PANOCCache, PANOCOptimizer},
    Optimizer, Problem, SolverError,
};

use self::{
    cost::{cost, cost_grad, ControlOutput},
    target::MpcTarget,
};

mod cost;
mod state;
mod target;

pub struct MpcConfig {
    /// Time resolution of the MPC
    pub dt: f64,
    /// Time horizon in seconds
    pub time_horizon: f64,
}

impl MpcConfig {
    pub(self) fn timesteps(&self) -> usize {
        (self.time_horizon / self.dt).ceil() as usize
    }
}

pub fn mpc(config: MpcConfig, targets: Vec<MpcTarget>, world: &WorldData) {
    let cost_f = |u: &[f64], c: &mut f64| -> Result<(), SolverError> {
        *c = cost(&config, &targets, world, u);
        Ok(())
    };
    let cost_df = |u: &[f64], grad: &mut [f64]| -> Result<(), SolverError> {
        cost_grad(&config, &targets, world, u, grad);
        Ok(())
    };

    let constraints = NoConstraints::new();
    let problem = Problem::new(&constraints, cost_df, cost_f);

    let n = 50;
    let lbfgs_memory = 10;
    let tolerance = 1e-6;
    let max_iters = 200;
    let mut panoc_cache = PANOCCache::new(n, tolerance, lbfgs_memory);

    let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

    let mut initial_u = ControlOutput::initialize(&world, config.timesteps());
    let status = panoc.solve(&mut initial_u).unwrap();
    tracing::info!("MPC solver status {status:?}");
}
