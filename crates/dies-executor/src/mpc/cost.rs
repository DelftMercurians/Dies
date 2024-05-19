use dies_core::WorldData;
use nalgebra::{Matrix1xX, Matrix2xX, MatrixView1xX, MatrixView2xX};

use super::{state::State, target::MpcTarget, MpcConfig};

pub type ControlOutputItem<'a> = (MatrixView2xX<'a, f64>, MatrixView1xX<'a, f64>);

/// Wraps the control output vector for easier access.
///
/// **WARNING**: The player _indices_ are **not** the same as the player _IDs_ used everywhere else.
pub struct ControlOutput {
    vel_u: Matrix2xX<f64>,
    w_u: Matrix1xX<f64>,
    num_players: usize,
    num_timesteps: usize,
}

impl ControlOutput {
    /// Create a new `ControlOutput` instance.
    ///
    /// # Panics
    ///
    /// Panics if the length of the control output vector is not equal to
    /// `num_players * num_timesteps * 3`.
    pub fn new(u: &[f64], num_players: usize, num_timesteps: usize) -> Self {
        // u layout: `([u_x, u_y] * num_players * num_timesteps) + ([u_w] * num_players * num_timesteps)`
        assert!(
            u.len() == num_players * num_timesteps * 3,
            "Length of control output vector does not match num_players * num_timesteps * 3"
        );
        let split = num_players * num_timesteps * 2;
        Self {
            vel_u: Matrix2xX::from_column_slice(&u[..split]),
            w_u: Matrix1xX::from_column_slice(&u[split..]),
            num_players,
            num_timesteps,
        }
    }

    /// Get a view of the player velocities for a specific timestep.
    pub fn velocities(&self, timestep: usize) -> MatrixView2xX<f64> {
        let start = timestep * self.num_players;
        self.vel_u.columns(start, self.num_players)
    }

    /// Get a view of the player angular velocities for a specific timestep.
    pub fn angular_velocities(&self, timestep: usize) -> MatrixView1xX<f64> {
        let start = timestep * self.num_players;
        self.w_u.columns(start, self.num_players)
    }

    /// Create initial control output using current velocites
    pub fn initialize(world: &WorldData, num_timesteps: usize) -> Vec<f64> {
        let num_players = world.own_players.len();
        let size = num_players * num_timesteps * 3;
        let mut initial_u = Vec::with_capacity(size);

        for timestep in 0..num_timesteps {
            for (player_idx, player) in world.own_players.iter().enumerate() {
                let pos_idx = (timestep * 2 * num_players) + player_idx;
                initial_u[pos_idx] = player.velocity.x;
                initial_u[pos_idx + 1] = player.velocity.y;
                let heading_idx =
                    (num_timesteps * 2 * num_players) + (timestep * num_players) + player_idx;
                initial_u[heading_idx] = player.angular_speed;
            }
        }

        initial_u
    }
}

pub fn cost(config: &MpcConfig, targets: &Vec<MpcTarget>, world: &WorldData, u: &[f64]) -> f64 {
    let num_timesteps = config.timesteps();
    let num_players = world.own_players.len();
    let mut state = State::new(world);
    let u = ControlOutput::new(u, num_players, num_timesteps);

    let mut cost = 0.0;
    for timestep in 0..num_timesteps {
        for (player_idx, target) in targets.iter().enumerate() {
            let position = state.position(player_idx);
            let heading = state.heading(player_idx);
            cost += target.cost(position, heading, state.ball_position())
        }

        let velocities = u.velocities(timestep);
        let angular_velocities = u.angular_velocities(timestep);
        state.step(config.dt, velocities, angular_velocities);
    }

    cost
}

pub fn cost_grad(
    config: &MpcConfig,
    targets: &Vec<MpcTarget>,
    world: &WorldData,
    u: &[f64],
    grad: &mut [f64],
) {
    let num_timesteps = config.timesteps();
    let num_players = world.own_players.len();
    let mut state = State::new(world);
    let u = ControlOutput::new(u, num_players, num_timesteps);

    for timestep in 0..num_timesteps {
        for (player_idx, target) in targets.iter().enumerate() {
            let position = state.position(player_idx);
            let heading = state.heading(player_idx);
            let (pos_grad, heading_grad) =
                target.cost_grad(position, heading, state.ball_position());

            let pos_idx = (timestep * 2 * num_players) + player_idx;
            grad[pos_idx] = pos_grad[0];
            grad[pos_idx + 1] = pos_grad[1];

            let heading_idx =
                (num_timesteps * 2 * num_players) + (timestep * num_players) + player_idx;
            grad[heading_idx] = heading_grad;
        }

        let velocities = u.velocities(timestep);
        let angular_velocities = u.angular_velocities(timestep);
        state.step(config.dt, velocities, angular_velocities);
    }
}
