use dies_core::{Vector2, TeamData, PlayerId};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use std::time::Instant;
use std::collections::HashMap;

#[derive(Clone)]
pub struct RobotState {
    pub id: PlayerId,
    pub position: Vector2,
    pub velocity: Vector2,
    pub target_position: Vector2,
    pub vel_limit: f64,
}

pub struct MPCController {
    field_bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
    last_solve_time_ms: f64,
    last_control_sequences: HashMap<PlayerId, Vec<Vector2>>,
    last_trajectories: HashMap<PlayerId, Vec<Vec<f64>>>, // Store full trajectory data [t, x, y, vx, vy]
}

impl MPCController {
    pub fn new() -> Self {
        Self {
            field_bounds: None,
            last_solve_time_ms: f64::NAN,
            last_control_sequences: HashMap::new(),
            last_trajectories: HashMap::new(),
        }
    }

    pub fn set_field_bounds(&mut self, world: &TeamData) {
        if let Some(geom) = &world.field_geom {
            let hw = geom.field_width / 2.0;
            let hl = geom.field_length / 2.0;
            self.field_bounds = Some((-hl, hl, -hw, hw));
        }
    }

    pub fn compute_batch_control(&mut self, robots: &[RobotState], world: &TeamData) -> HashMap<PlayerId, Vector2> {
        if robots.is_empty() {
            return HashMap::new();
        }

        match self.solve_batch_mpc_jax(robots, world) {
            Ok(controls) => controls,
            Err(e) => {
                log::warn!("JAX batch MPC failed: {}, falling back to simple control", e);
                self.last_solve_time_ms = 0.0; // Mark as failed
                // Fallback: simple proportional control for each robot
                let mut fallback_controls = HashMap::new();
                for robot in robots {
                    let error = robot.target_position - robot.position;
                    let desired_vel = error.normalize() * robot.vel_limit.min(error.magnitude() * 2.0);
                    let control = desired_vel.cap_magnitude(robot.vel_limit);
                    fallback_controls.insert(robot.id, control);

                    // Store fallback as repeated control sequence
                    let fallback_sequence = vec![control; 10]; // Use a reasonable default
                    self.last_control_sequences.insert(robot.id, fallback_sequence);
                }
                fallback_controls
            }
        }
    }

    pub fn last_solve_time_ms(&self) -> f64 {
        self.last_solve_time_ms
    }

    pub fn get_trajectories(&self) -> &HashMap<PlayerId, Vec<Vec<f64>>> {
        &self.last_trajectories
    }

    fn solve_batch_mpc_jax(&mut self, robots: &[RobotState], world: &TeamData) -> Result<HashMap<PlayerId, Vector2>, PyErr> {
        let start_time = Instant::now();

        Python::with_gil(|py| {
            // Set up stdout capture
            let sys = PyModule::import(py, "sys")?;
            let io = PyModule::import(py, "io")?;
            let stdout_capture = io.call_method0("StringIO")?;
            let original_stdout = sys.getattr("stdout")?;
            sys.setattr("stdout", &stdout_capture)?;

            // Add the mpc_jax directory to Python path
            let sys_path = sys.getattr("path")?;
            sys_path.call_method1("insert", (0, "./.venv/lib/python3.12/site-packages"))?;
            sys_path.call_method1("insert", (0, "."))?;

            // Import the mpc_jax module and get constants
            let mpc_module = PyModule::import(py, "mpc_jax")?;
            let common_module = PyModule::import(py, "mpc_jax.common")?;
            let control_horizon: usize = common_module.getattr("CONTROL_HORIZON")?.extract()?;

            // Import numpy for array creation
            let np = PyModule::import(py, "numpy")?;

            // Prepare batched robot data
            let n_robots = robots.len();
            let mut initial_positions = Vec::new();
            let mut initial_velocities = Vec::new();
            let mut target_positions = Vec::new();
            let mut vel_limits = Vec::new();
            let mut last_controls_data = Vec::new();

            for robot in robots {
                initial_positions.extend_from_slice(&[robot.position.x, robot.position.y]);
                initial_velocities.extend_from_slice(&[robot.velocity.x, robot.velocity.y]);
                target_positions.extend_from_slice(&[robot.target_position.x, robot.target_position.y]);
                vel_limits.push(robot.vel_limit);

                // Get last control sequence for this robot, or zeros if not available
                let last_sequence = self.last_control_sequences.get(&robot.id)
                    .cloned()
                    .unwrap_or_else(|| vec![Vector2::new(0.0, 0.0); control_horizon]);

                for control in &last_sequence {
                    last_controls_data.extend_from_slice(&[control.x, control.y]);
                }
            }

            // Convert to numpy arrays with proper shapes
            let initial_pos_array = np.call_method1("array", (initial_positions,))?.call_method1("reshape", (n_robots, 2))?;
            let initial_vel_array = np.call_method1("array", (initial_velocities,))?.call_method1("reshape", (n_robots, 2))?;
            let target_pos_array = np.call_method1("array", (target_positions,))?.call_method1("reshape", (n_robots, 2))?;
            let vel_limits_array = np.call_method1("array", (vel_limits,))?;
            let last_controls_array = np.call_method1("array", (last_controls_data,))?.call_method1("reshape", (n_robots, control_horizon, 2))?;

            // Prepare obstacles (only other robots, NOT the ball)
            let mut obstacles_data = Vec::new();

            // Add opponent robots as obstacles
            for opp_player in &world.opp_players {
                obstacles_data.extend_from_slice(&[opp_player.position.x, opp_player.position.y]);
            }

            // Convert obstacles to numpy array
            let obstacles = if obstacles_data.is_empty() {
                np.call_method1("zeros", ((0, 2),))?
            } else {
                let n = obstacles_data.len();
                let obstacles_array = np.call_method1("array", (obstacles_data,))?
                    .call_method1("reshape", ((n / 2) as i32, 2))?;
                obstacles_array
            };

            // Handle ball separately with proper position
            let ball_pos = if let Some(ball) = &world.ball {
                Some(np.call_method1("array", (vec![ball.position.xy().x, ball.position.xy().y],))?)
            } else {
                None
            };

            // Prepare field bounds
            let field_bounds = match self.field_bounds {
                Some((min_x, max_x, min_y, max_y)) => {
                    Some(np.call_method1("array", (vec![min_x, max_x, min_y, max_y],))?)
                }
                None => None,
            };

            // Prepare last trajectory data with dynamic horizon length
            let mut last_traj_data = Vec::new();
            let mut dynamic_horizon = control_horizon * 5 + 1; // Default to control_horizon

            for robot in robots {
                if let Some(traj) = self.last_trajectories.get(&robot.id) {
                    // Use the actual trajectory length for dynamic horizon
                    if !traj.is_empty() {
                        dynamic_horizon = traj.len();
                    }

                    // Use stored trajectory data directly
                    for i in 0..dynamic_horizon {
                        if i < traj.len() && traj[i].len() >= 5 {
                            last_traj_data.extend_from_slice(&traj[i]);
                        } else {
                            // Pad with last available value or zeros if no trajectory exists
                            let last_point = if !traj.is_empty() && !traj.last().unwrap().is_empty() {
                                traj.last().unwrap().clone()
                            } else {
                                vec![i as f64 * world.dt, 0.0, 0.0, 0.0, 0.0]
                            };
                            last_traj_data.extend_from_slice(&last_point);
                        }
                    }
                } else {
                    // No previous trajectory, use zeros with proper time values
                    for i in 0..dynamic_horizon {
                        let t = i as f64 * world.dt;
                        last_traj_data.extend_from_slice(&[t, 0.0, 0.0, 0.0, 0.0]);
                    }
                }
            }

            let last_traj_array = if last_traj_data.is_empty() {
                py.None()
            } else {
                np.call_method1("array", (last_traj_data,))?.call_method1("reshape", (n_robots, dynamic_horizon, 5))?.into()
            };

            let dt_value = world.dt;

            // Call the JAX batch solve_mpc function with ball passed separately
            let solve_mpc_batch = mpc_module.getattr("solve_mpc_tbwrap")?;
            let result = solve_mpc_batch.call1((
                initial_pos_array,
                initial_vel_array,
                target_pos_array,
                obstacles,
                ball_pos,
                field_bounds,
                vel_limits_array,
                last_controls_array,
                dt_value,
            ))?;

            // Extract the result - it's a tuple of (controls, trajectories)
            let result_tuple: (Py<PyAny>, Py<PyAny>) = result.extract()?;
            let controls_py = result_tuple.0;
            let trajectories_py = result_tuple.1;

            // Convert controls to Rust data
            let controls_list = controls_py.bind(py).call_method0(pyo3::intern!(py, "tolist"))?;
            let control_data: Vec<Vec<Vec<f64>>> = controls_list.extract()?;

            // Convert trajectories to Rust data
            let trajectories_list = trajectories_py.bind(py).call_method0(pyo3::intern!(py, "tolist"))?;
            let trajectory_data: Vec<Vec<Vec<f64>>> = trajectories_list.extract()?;

            // Restore stdout and capture any prints
            sys.setattr("stdout", original_stdout)?;
            let captured_output: String = stdout_capture.call_method0(pyo3::intern!(py, "getvalue"))?.extract()?;

            // Calculate timing
            let duration = start_time.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;

            if self.last_solve_time_ms.is_nan() {
                self.last_solve_time_ms = duration_ms;
            }
            if self.last_solve_time_ms < duration_ms / 4.0 {
                log::warn!("MPC abnormal timing: usually its around {:.1} but got {:.1}", self.last_solve_time_ms, duration_ms);
            }
            self.last_solve_time_ms = duration_ms * 0.05 + self.last_solve_time_ms * 0.95;

            // Output any Python prints using dies_core::debug
            if !captured_output.trim().is_empty() {
                dies_core::debug_string("mpc.python_output", captured_output.trim());
                // Also log to Rust log for visibility
                log::info!("MPC Python output: {}", captured_output.trim());
            }

            // Convert results back to HashMap and store full sequences
            let mut controls = HashMap::new();
            for (i, robot) in robots.iter().enumerate() {
                // Handle control sequences
                if let Some(control_sequence) = control_data.get(i) {
                    if !control_sequence.is_empty() {
                        // Extract first control for immediate use
                        if let Some(first_control) = control_sequence.get(0) {
                            if first_control.len() >= 2 {
                                controls.insert(robot.id, Vector2::new(first_control[0], first_control[1]));
                            }
                        }

                        // Store full control sequence for continuity
                        let sequence: Vec<Vector2> = control_sequence.iter()
                            .filter_map(|control| {
                                if control.len() >= 2 {
                                    Some(Vector2::new(control[0], control[1]))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !sequence.is_empty() {
                            self.last_control_sequences.insert(robot.id, sequence);
                        }
                    }
                }

                // Handle trajectory data
                if let Some(trajectory_sequence) = trajectory_data.get(i) {
                    if !trajectory_sequence.is_empty() {
                        // Store full trajectory data [t, x, y, vx, vy] for each point
                        let traj_points: Vec<Vec<f64>> = trajectory_sequence.iter()
                            .filter_map(|point| {
                                if point.len() >= 5 {
                                    Some(point.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !traj_points.is_empty() {
                            self.last_trajectories.insert(robot.id, traj_points);
                        }
                    }
                }
            }

            Ok(controls)
        })
    }

}

impl Default for MPCController {
    fn default() -> Self {
        Self::new()
    }
}
