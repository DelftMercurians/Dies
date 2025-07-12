use dies_core::{PlayerId, TeamData, Vector2};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;

#[derive(Clone)]
pub struct RobotState {
    pub id: PlayerId,
    pub position: Vector2,
    pub velocity: Vector2,
    pub target_position: Vector2,
    pub vel_limit: f64,
}

#[derive(Clone)]
struct MPCControllerState {
    field_bounds: Option<(f64, f64, f64, f64)>,
    last_control_sequences: HashMap<PlayerId, Vec<Vector2>>,
    last_trajectories: HashMap<PlayerId, Vec<Vec<f64>>>,
    last_solve_time_ms: f64,
}

struct MPCRequest {
    robots: Vec<RobotState>,
    world: TeamData,
    controller_state: MPCControllerState,
    controllable_mask: Vec<bool>,
}

struct MPCResponse {
    controls: HashMap<PlayerId, Vector2>,
    control_sequences: HashMap<PlayerId, Vec<Vector2>>,
    updated_state: MPCControllerState,
}

pub struct MPCController {
    field_bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
    last_solve_time_ms: f64,
    last_control_sequences: HashMap<PlayerId, Vec<Vector2>>,
    last_trajectories: HashMap<PlayerId, Vec<Vec<f64>>>, // Store full trajectory data [t, x, y, vx, vy]

    // Thread communication
    request_sender: mpsc::SyncSender<MPCRequest>,
    response_receiver: mpsc::Receiver<MPCResponse>,
    last_mpc_result: Option<HashMap<PlayerId, Vector2>>,
    last_mpc_time: Instant,
    request_pending: bool,
    _thread_handle: thread::JoinHandle<()>,
}

impl MPCController {
    pub fn new() -> Self {
        let (request_sender, request_receiver) = mpsc::sync_channel::<MPCRequest>(2);
        let (response_sender, response_receiver) = mpsc::sync_channel::<MPCResponse>(2);

        let thread_handle = thread::Builder::new()
            .name("mpc-worker".to_string())
            .spawn(move || {
                Self::mpc_worker_thread(request_receiver, response_sender);
            })
            .expect("Failed to spawn MPC worker thread");

        Self {
            field_bounds: None,
            last_solve_time_ms: f64::NAN,
            last_control_sequences: HashMap::new(),
            last_trajectories: HashMap::new(),
            request_sender,
            response_receiver,
            last_mpc_result: None,
            last_mpc_time: Instant::now(),
            request_pending: false,
            _thread_handle: thread_handle,
        }
    }

    pub fn set_field_bounds(&mut self, world: &TeamData) {
        if let Some(geom) = &world.field_geom {
            let hw = geom.field_width / 2.0;
            let hl = geom.field_length / 2.0;
            self.field_bounds = Some((-hl, hl, -hw, hw));
        }
    }

    pub fn compute_batch_control(
        &mut self,
        robots: &[RobotState],
        world: &TeamData,
        controllable_mask: Option<&[bool]>,
    ) -> HashMap<PlayerId, Vector2> {
        if robots.is_empty() {
            return HashMap::new();
        }

        // Check for completed results from worker thread
        while let Ok(response) = self.response_receiver.try_recv() {
            // Update controller state with results from background thread
            self.last_control_sequences = response.updated_state.last_control_sequences;
            self.last_trajectories = response.updated_state.last_trajectories;
            self.last_solve_time_ms = response.updated_state.last_solve_time_ms;

            self.last_mpc_result = Some(response.controls.clone());
            self.last_mpc_time = Instant::now();
            self.request_pending = false; // Clear the pending flag when we get a response
            return response.controls;
        }

        // Check if we should use last result or send new request
        let elapsed = self.last_mpc_time.elapsed().as_millis();

        if elapsed > 150 {
            // If delay is too long, fall back to MTP (return empty HashMap)
            log::warn!("mpc took too long, switching back to the mtp");
            return HashMap::new();
        }

        // If delay is acceptable, always try to send new request unless already pending
        if !self.request_pending {
            // Send new request to worker thread (non-blocking)
            let controller_state = MPCControllerState {
                field_bounds: self.field_bounds,
                last_control_sequences: self.last_control_sequences.clone(),
                last_trajectories: self.last_trajectories.clone(),
                last_solve_time_ms: self.last_solve_time_ms,
            };

            let request = MPCRequest {
                robots: robots.to_vec(),
                world: world.clone(),
                controller_state,
                controllable_mask: controllable_mask.unwrap_or(&vec![true; robots.len()]).to_vec(),
            };

            if let Err(e) = self.request_sender.try_send(request) {
                log::warn!("mpc thread is having troubles: {}", e);
            } else {
                self.request_pending = true; // Mark request as pending
            }
        }

        // Return interpolated control based on elapsed time or empty (let MTP handle it)
        if let Some(ref last_result) = self.last_mpc_result {
            self.interpolate_controls_at_time(elapsed as f64)
        } else {
            HashMap::new()
        }
    }

    pub fn last_solve_time_ms(&self) -> f64 {
        self.last_solve_time_ms
    }

    pub fn get_trajectories(&self) -> &HashMap<PlayerId, Vec<Vec<f64>>> {
        &self.last_trajectories
    }

    fn interpolate_controls_at_time(&self, elapsed_ms: f64) -> HashMap<PlayerId, Vector2> {
        let mut interpolated_controls = HashMap::new();

        // Target time step is 80ms, so calculate which control step we should interpolate to
        let target_time_step = 50.0; // ms
        let step_index = (elapsed_ms + 30.0) / target_time_step;
        let step_floor = step_index.floor() as usize;
        let step_frac = step_index.fract();

        for (robot_id, control_sequence) in &self.last_control_sequences {
            if control_sequence.len() > step_floor {
                let current_control = control_sequence[step_floor];

                if control_sequence.len() > step_floor + 1 && step_frac > 0.0 {
                    // Interpolate between current and next control
                    let next_control = control_sequence[step_floor + 1];
                    let interpolated = Vector2::new(
                        current_control.x * (1.0 - step_frac) + next_control.x * step_frac,
                        current_control.y * (1.0 - step_frac) + next_control.y * step_frac,
                    );
                    interpolated_controls.insert(*robot_id, interpolated);
                } else {
                    // Use current control if no next step available
                    interpolated_controls.insert(*robot_id, current_control);
                }
            }
        }

        interpolated_controls
    }

    fn should_fallback_to_mtp(&self) -> bool {
        // Check if we should fallback - either no last result or timeout exceeded
        if self.last_mpc_result.is_none() {
            return true;
        }

        let elapsed = self.last_mpc_time.elapsed().as_millis();
        elapsed > 150
    }

    fn mpc_worker_thread(
        request_receiver: mpsc::Receiver<MPCRequest>,
        response_sender: mpsc::SyncSender<MPCResponse>,
    ) {
        loop {
            let request = match request_receiver.try_recv() {
                Ok(req) => req,
                Err(mpsc::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
                Err(mpsc::TryRecvError::Disconnected) => break,
            };
            let mut controller_state = request.controller_state;

            // Keep an untouched copy for the “panic fallback” path.
            let state_fallback = controller_state.clone();

            let (controls, updated_state) = match Self::solve_batch_mpc_jax_sync(
                &request.robots,
                &request.world,
                &mut controller_state,
                &request.controllable_mask,
            ) {
                Ok(controls) => (controls, controller_state),
                Err(e) => {
                    log::warn!("JAX batch-MPC failed: {e}; returning empty result");
                    (HashMap::new(), state_fallback)
                }
            };

            // Extract control sequences from updated state before moving it
            let control_sequences = updated_state.last_control_sequences.clone();

            let response = MPCResponse {
                controls: controls.clone(),
                control_sequences,
                updated_state,
            };

            if let Err(e) = response_sender.try_send(response) {
                log::error!("Failed to send MPC response: {e}");
                break;
            }
        }
    }

    fn solve_batch_mpc_jax_sync(
        robots: &[RobotState],
        world: &TeamData,
        controller_state: &mut MPCControllerState,
        controllable_mask: &[bool],
    ) -> Result<HashMap<PlayerId, Vector2>, PyErr> {
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
            sys_path.call_method1("insert", (0, "./.venv/lib/python3.13/site-packages"))?;
            sys_path.call_method1("insert", (0, "./mpc_jax"))?;
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
                target_positions
                    .extend_from_slice(&[robot.target_position.x, robot.target_position.y]);
                vel_limits.push(robot.vel_limit);

                // Get last control sequence for this robot, or zeros if not available
                let last_sequence = controller_state
                    .last_control_sequences
                    .get(&robot.id)
                    .cloned()
                    .unwrap_or_else(|| vec![Vector2::new(0.0, 0.0); control_horizon]);

                for control in &last_sequence {
                    last_controls_data.extend_from_slice(&[control.x, control.y]);
                }
            }

            // Convert to numpy arrays with proper shapes
            let initial_pos_array = np
                .call_method1("array", (initial_positions,))?
                .call_method1("reshape", (n_robots, 2))?;
            let initial_vel_array = np
                .call_method1("array", (initial_velocities,))?
                .call_method1("reshape", (n_robots, 2))?;
            let target_pos_array = np
                .call_method1("array", (target_positions,))?
                .call_method1("reshape", (n_robots, 2))?;
            let vel_limits_array = np.call_method1("array", (vel_limits,))?;
            let last_controls_array = np
                .call_method1("array", (last_controls_data,))?
                .call_method1("reshape", (n_robots, control_horizon, 2))?;

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
                let obstacles_array = np
                    .call_method1("array", (obstacles_data,))?
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
            let field_bounds = match controller_state.field_bounds {
                Some((min_x, max_x, min_y, max_y)) => {
                    Some(np.call_method1("array", (vec![min_x, max_x, min_y, max_y],))?)
                }
                None => None,
            };

            // Prepare field geometry [field_length, field_width, penalty_area_depth, penalty_area_width]
            let field_geometry = if let Some(geom) = &world.field_geom {
                Some(np.call_method1("array", (vec![geom.field_length, geom.field_width, geom.penalty_area_depth, geom.penalty_area_width],))?)
            } else {
                None
            };

            let dt_value = world.dt;

            // Prepare controllable mask as numpy array
            let controllable_mask_data: Vec<i32> = controllable_mask.iter().map(|&b| if b { 1 } else { 0 }).collect();
            let controllable_mask_array = np.call_method1("array", (controllable_mask_data,))?;

            // Call the JAX batch solve_mpc function with field geometry
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
                field_geometry,
                controllable_mask_array
            ))?;

            // Extract the result - it's a tuple of (controls, trajectories)
            let result_tuple: (Py<PyAny>, Py<PyAny>) = result.extract()?;
            let controls_py = result_tuple.0;
            let trajectories_py = result_tuple.1;

            // Convert controls to Rust data
            let controls_list = controls_py
                .bind(py)
                .call_method0(pyo3::intern!(py, "tolist"))?;
            let control_data: Vec<Vec<Vec<f64>>> = controls_list.extract()?;

            // Convert trajectories to Rust data
            let trajectories_list = trajectories_py
                .bind(py)
                .call_method0(pyo3::intern!(py, "tolist"))?;
            let trajectory_data: Vec<Vec<Vec<f64>>> = trajectories_list.extract()?;

            // Restore stdout and capture any prints
            sys.setattr("stdout", original_stdout)?;
            let captured_output: String = stdout_capture
                .call_method0(pyo3::intern!(py, "getvalue"))?
                .extract()?;

            // Calculate timing
            let duration = start_time.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;

            if controller_state.last_solve_time_ms.is_nan() {
                controller_state.last_solve_time_ms = duration_ms;
            }
            if controller_state.last_solve_time_ms < duration_ms / 4.0 {
                log::warn!(
                    "MPC abnormal timing: usually its around {:.1} but got {:.1}",
                    controller_state.last_solve_time_ms,
                    duration_ms
                );
            }
            controller_state.last_solve_time_ms =
                duration_ms * 0.05 + controller_state.last_solve_time_ms * 0.95;

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
                                controls.insert(
                                    robot.id,
                                    Vector2::new(first_control[0], first_control[1]),
                                );
                            }
                        }

                        // Store full control sequence for continuity
                        let sequence: Vec<Vector2> = control_sequence
                            .iter()
                            .filter_map(|control| {
                                if control.len() >= 2 {
                                    Some(Vector2::new(control[0], control[1]))
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !sequence.is_empty() {
                            controller_state
                                .last_control_sequences
                                .insert(robot.id, sequence);
                        }
                    }
                }

                // Handle trajectory data
                if let Some(trajectory_sequence) = trajectory_data.get(i) {
                    if !trajectory_sequence.is_empty() {
                        // Store full trajectory data [t, x, y, vx, vy] for each point
                        let traj_points: Vec<Vec<f64>> = trajectory_sequence
                            .iter()
                            .filter_map(|point| {
                                if point.len() >= 5 {
                                    Some(point.clone())
                                } else {
                                    None
                                }
                            })
                            .collect();

                        if !traj_points.is_empty() {
                            controller_state
                                .last_trajectories
                                .insert(robot.id, traj_points);
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
