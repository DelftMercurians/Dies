use dies_core::WorldData;
use optimization_engine::{
    constraints::NoConstraints,
    panoc::{PANOCCache, PANOCOptimizer},
    Optimizer, Problem, SolverError,
};

use crate::mpc::control_output::initialize_u;

use self::{
    cost::{cost, cost_grad},
    target::MpcTarget,
};

mod control_output;
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

pub struct MpcSolver {
    config: MpcConfig,
    num_players: usize,
    cache: PANOCCache,
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

    let n = config.timesteps() * 3 * world.own_players.len();
    let lbfgs_memory = 100;
    let tolerance = 1e-3;
    let max_iters = 2000;
    let mut panoc_cache = PANOCCache::new(n, tolerance, lbfgs_memory);

    let mut panoc = PANOCOptimizer::new(problem, &mut panoc_cache).with_max_iter(max_iters);

    let mut initial_u = initialize_u(&world, config.timesteps());
    let status = panoc.solve(&mut initial_u).unwrap();
    println!("MPC solver status {status:?}");

    println!("Optimal control input: {initial_u:?}");
}

#[cfg(test)]
mod test {
    use self::target::{HeadingTarget, PositionTarget};

    use super::*;
    use dies_core::{BallData, GameState, GameStateData, PlayerData, PlayerId, Vector2, Vector3};

    #[test]
    fn test_mpc() {
        let config = MpcConfig {
            dt: 0.1,
            time_horizon: 1.0,
        };

        // Create own players
        let num_players = 4;
        let mut world = WorldData {
            own_players: Vec::with_capacity(num_players),
            ..Default::default()
        };
        let mut targets = Vec::new();
        for i in 0..num_players {
            world.own_players.push(PlayerData {
                id: PlayerId::new(i),
                position: Vector2::new((i + 1) as f64 * 100.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                raw_position: Vector2::new(0.0, 0.0),
                orientation: 0.0,
                angular_speed: 0.0,
                timestamp: 0.0,
            });

            targets.push(MpcTarget {
                heading_target: HeadingTarget::None,
                position_target: PositionTarget::ConstantPosition(Vector2::new(0.0, 0.0)),
            });
        }

        mpc(config, targets, &world);
    }
}
