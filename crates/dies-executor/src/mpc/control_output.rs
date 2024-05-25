use dies_core::{Vector2, WorldData};

use super::MpcConfig;

pub struct ControlOutputItem {
    pub velocity: Vector2,
    pub angular_velocity: f64,
}

/// Wraps the control output vector for easier access.
///
/// **WARNING**: The player _indices_ are **not** the same as the player _IDs_ used everywhere else.
pub struct ControlOutput<'a> {
    /// Control output vector for player velocities and angular velocities.
    ///
    /// Layout: `(([u_x, u_y, u_w] * num_players) * num_timesteps)`
    pub(crate) u: &'a [f64],
    pub(crate) num_players: usize,
    pub(crate) num_timesteps: usize,
}

impl<'a> ControlOutput<'a> {
    /// Create a new `ControlOutput` instance.
    ///
    /// # Panics
    ///
    /// Panics if the length of the control output vector is not equal to
    /// `num_players * num_timesteps * 3`.
    pub fn new(u: &'a [f64], num_players: usize, num_timesteps: usize) -> Self {
        assert!(
            u.len() == num_players * num_timesteps * 3,
            "Length of control output vector does not match num_players * num_timesteps * 3"
        );
        Self {
            u,
            num_players,
            num_timesteps,
        }
    }

    /// Get the control output for a specific player at a specific timestep.
    pub fn player(&self, timestep: usize, player_idx: usize) -> ControlOutputItem {
        let stating_idx = (timestep * 3 * self.num_players) + (player_idx * 3);
        let velocity = Vector2::new(self.u[stating_idx], self.u[stating_idx + 1]);
        let angular_velocity = self.u[stating_idx + 2];
        ControlOutputItem {
            velocity,
            angular_velocity,
        }
    }

    pub fn timestep(&self, timestep: usize) -> Vec<ControlOutputItem> {
        (0..self.num_players)
            .map(|player_idx| self.player(timestep, player_idx))
            .collect()
    }
}

/// Create initial control output using current velocites
pub fn initialize_u(world: &WorldData, num_timesteps: usize) -> Vec<f64> {
    let num_players = world.own_players.len();
    let size = num_players * num_timesteps * 3;
    let mut initial_u = vec![0.0; size];

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
