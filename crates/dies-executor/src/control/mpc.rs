use dies_core::{Vector2, WorldData, PlayerId};
use autograd as ag;
use ag::tensor::Constant;

const ROBOT_RADIUS: f64 = 90.0; // mm
const FIELD_BOUNDARY_MARGIN: f64 = 100.0; // mm
const COLLISION_PENALTY_RADIUS: f64 = 200.0; // mm
const PREDICTION_HORIZON: usize = 10;
const CONTROL_HORIZON: usize = 8;
const DT: f64 = 0.05; // 200ms time step

pub struct MPCController {
    horizon: usize,
    control_horizon: usize,
    dt: f64,
    target_position: Vector2,
    obstacles: Vec<Vector2>,
    field_bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
}

impl MPCController {
    pub fn new() -> Self {
        Self {
            horizon: PREDICTION_HORIZON,
            control_horizon: CONTROL_HORIZON,
            dt: DT,
            target_position: Vector2::zeros(),
            obstacles: Vec::new(),
            field_bounds: None,
        }
    }

    pub fn set_target(&mut self, target: Vector2) {
        self.target_position = target;
    }

    pub fn set_obstacles(&mut self, obstacles: &[Vector2]) {
        self.obstacles = obstacles.to_vec();
    }

    pub fn set_field_bounds(&mut self, world: &WorldData) {
        if let Some(geom) = &world.field_geom {
            let hw = geom.field_width / 2.0 - FIELD_BOUNDARY_MARGIN;
            let hl = geom.field_length / 2.0 - FIELD_BOUNDARY_MARGIN;
            self.field_bounds = Some((-hl, hl, -hw, hw));
        }
    }

    pub fn compute_control(&mut self, current_pos: Vector2, current_vel: Vector2, vel_limit: f64) -> Vector2 {
        let num_vars = self.control_horizon * 2; // vx, vy for each control step

        // Initial guess: maintain current velocity
        let mut u = vec![0.0; num_vars];
        for i in 0..self.control_horizon {
            u[i * 2] = (-current_vel.x).min(vel_limit).max(-vel_limit);
            u[i * 2 + 1] = (-current_vel.y).min(vel_limit).max(-vel_limit);
        }

        // Use gradient descent to optimize the control sequence
        let learning_rate = 0.01;
        let max_iterations = 20;

        for _iter in 0..max_iterations {
            ag::with(|g: &mut ag::Graph<_>| {
                // Create placeholders for control inputs
                let u_placeholders: Vec<_> = (0..num_vars).map(|_| g.placeholder(&[])).collect();

                // Simulate forward and compute cost
                let mut cost = g.scalar(0.0);
                let mut pos_x = g.scalar(current_pos.x);
                let mut pos_y = g.scalar(current_pos.y);

                // Forward simulation over prediction horizon
                for k in 0..self.horizon {
                    let control_idx = (k.min(self.control_horizon - 1)) * 2;
                    let vel_x = if k < self.control_horizon {
                        u_placeholders[control_idx]
                    } else {
                        g.scalar(0.0)
                    };
                    let vel_y = if k < self.control_horizon {
                        u_placeholders[control_idx + 1]
                    } else {
                        g.scalar(0.0)
                    };

                    // Euler integration
                    pos_x = pos_x + vel_x * self.dt;
                    pos_y = pos_y + vel_y * self.dt;

                    // Distance to target cost
                    let target_x = g.scalar(self.target_position.x);
                    let target_y = g.scalar(self.target_position.y);
                    let dx = pos_x - target_x;
                    let dy = pos_y - target_y;
                    let dist_sq = dx * dx + dy * dy;
                    cost = cost + dist_sq * 10.0;

                    /*
                    // Collision avoidance cost
                    for &obstacle in &self.obstacles {
                        let obs_x = g.scalar(obstacle.x);
                        let obs_y = g.scalar(obstacle.y);
                        let obs_dx = pos_x - obs_x;
                        let obs_dy = pos_y - obs_y;
                        let obs_dist_sq = obs_dx * obs_dx + obs_dy * obs_dy;
                        let penalty_radius_sq = g.scalar(COLLISION_PENALTY_RADIUS * COLLISION_PENALTY_RADIUS);

                        // Soft constraint: penalty grows as we get closer
                        let penalty = penalty_radius_sq / (obs_dist_sq + g.scalar(1.0));
                        cost = cost + penalty * 1000.0;
                    }

                    // Field boundary cost
                    if let Some((min_x, max_x, min_y, max_y)) = self.field_bounds {
                        let min_x_var = g.scalar(min_x);
                        let max_x_var = g.scalar(max_x);
                        let min_y_var = g.scalar(min_y);
                        let max_y_var = g.scalar(max_y);

                        // Boundary penalties using maximum for ReLU-like behavior
                        let zero = g.constant(ag::ndarray::arr0(0.0));
                        let x_lower_violation = g.maximum(min_x_var - pos_x, zero);
                        let x_upper_violation = g.maximum(pos_x - max_x_var, zero);
                        let y_lower_violation = g.maximum(min_y_var - pos_y, zero);
                        let y_upper_violation = g.maximum(pos_y - max_y_var, zero);

                        cost = cost + (x_lower_violation * x_lower_violation + x_upper_violation * x_upper_violation +
                                     y_lower_violation * y_lower_violation + y_upper_violation * y_upper_violation) * 100.0;
                    }
                    */

                    // Control effort penalty
                    if k < self.control_horizon {
                        let fk = k as f64;
                        cost = cost + (vel_x * vel_x + vel_y * vel_y) * 0.01 * g.scalar(fk * fk);
                    }
                }

                // Velocity constraint penalty
                for i in 0..self.control_horizon {
                    let vx = u_placeholders[i * 2];
                    let vy = u_placeholders[i * 2 + 1];
                    let speed_sq = vx * vx + vy * vy;
                    let max_speed_sq = g.scalar(vel_limit * vel_limit);
                    let zero = g.constant(ag::ndarray::arr0(0.0));
                    let speed_penalty = g.maximum(speed_sq - max_speed_sq, zero);
                    cost = cost + speed_penalty * 1000.0;
                }

                // Compute gradients with respect to control inputs
                let grads = g.grad(&[cost], &u_placeholders);

                // Create feed for current control values
                let u_arrays: Vec<_> = u.iter().map(|&val| ag::ndarray::arr0(val)).collect();
                let mut feed = Vec::new();
                for i in 0..num_vars {
                    feed.push(u_placeholders[i].given(u_arrays[i].view()));
                }

                // Update control inputs using gradient descent
                for (i, grad) in grads.iter().enumerate() {
                    if let Ok(grad_val) = grad.eval(&feed) {
                        let grad_scalar = grad_val.into_dimensionality::<ag::ndarray::Ix0>().unwrap()[()];
                        u[i] -= learning_rate * grad_scalar;
                        if i == 0 {
                            println!("u[0] = {}", u[0]);
                        }
                        // Simple velocity bound enforcement
                        u[i] = u[i].min(vel_limit).max(-vel_limit);
                    }
                }
            });
        }

        // Return the first control action
        Vector2::new(u[0], u[1])
    }

    pub fn update_obstacles_from_world(&mut self, world: &WorldData, current_player_id: PlayerId) {
        let mut obstacles = Vec::new();

        // Add other robots as obstacles
        for player in world.own_players.iter().chain(world.opp_players.iter()) {
            if player.id != current_player_id {
                obstacles.push(player.position);
            }
        }

        // Add ball as obstacle if present
        if let Some(ball) = &world.ball {
            obstacles.push(ball.position.xy());
        }

        self.set_obstacles(&obstacles);
    }
}

impl Default for MPCController {
    fn default() -> Self {
        Self::new()
    }
}
