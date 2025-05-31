use dies_core::{Vector2, WorldData, PlayerId};
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use std::time::Instant;

pub struct MPCController {
    target_position: Vector2,
    obstacles: Vec<Vector2>,
    field_bounds: Option<(f64, f64, f64, f64)>, // (min_x, max_x, min_y, max_y)
    last_solve_time_ms: f64,
}

impl MPCController {
    pub fn new() -> Self {
        Self {
            target_position: Vector2::zeros(),
            obstacles: Vec::new(),
            field_bounds: None,
            last_solve_time_ms: f64::NAN,
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
            let hw = geom.field_width / 2.0;
            let hl = geom.field_length / 2.0;
            self.field_bounds = Some((-hl, hl, -hw, hw));
        }
    }

    pub fn compute_control(&mut self, current_pos: Vector2, current_vel: Vector2, vel_limit: f64) -> Vector2 {
        match self.solve_mpc_jax(current_pos, current_vel, vel_limit) {
            Ok(control) => control,
            Err(e) => {
                log::warn!("JAX MPC failed: {}, falling back to simple control", e);
                self.last_solve_time_ms = 0.0; // Mark as failed
                // Fallback: simple proportional control
                let error = self.target_position - current_pos;
                let desired_vel = error.normalize() * vel_limit.min(error.magnitude() * 2.0);
                desired_vel.cap_magnitude(vel_limit)
            }
        }
    }

    pub fn last_solve_time_ms(&self) -> f64 {
        self.last_solve_time_ms
    }

    fn solve_mpc_jax(&mut self, current_pos: Vector2, current_vel: Vector2, vel_limit: f64) -> Result<Vector2, PyErr> {
        let start_time = Instant::now();

        Python::with_gil(|py| {
            // Set up stdout capture
            let sys = PyModule::import_bound(py, "sys")?;
            let io = PyModule::import_bound(py, "io")?;
            let stdout_capture = io.call_method0("StringIO")?;
            let original_stdout = sys.getattr("stdout")?;
            sys.setattr("stdout", &stdout_capture)?;

            // Add the mpc_jax directory to Python path
            let sys_path = sys.getattr("path")?;
            sys_path.call_method1("insert", (0, "."))?;

            // Import the mpc_jax module
            let mpc_module = PyModule::import_bound(py, "mpc_jax")?;

            // Import numpy for array creation
            let np = PyModule::import_bound(py, "numpy")?;

            // Prepare input data as numpy arrays
            let initial_pos = np.call_method1("array", (vec![current_pos.x, current_pos.y],))?;
            let initial_vel = np.call_method1("array", (vec![current_vel.x, current_vel.y],))?;
            let target_pos = np.call_method1("array", (vec![self.target_position.x, self.target_position.y],))?;

            // Prepare obstacles
            let obstacles = if self.obstacles.is_empty() {
                np.call_method1("zeros", ((0, 2),))?
            } else {
                let obstacles_data: Vec<Vec<f64>> = self.obstacles.iter()
                    .map(|obs| vec![obs.x, obs.y])
                    .collect();
                np.call_method1("array", (obstacles_data,))?
            };

            // Prepare field bounds
            let field_bounds = match self.field_bounds {
                Some((min_x, max_x, min_y, max_y)) => {
                    Some(np.call_method1("array", (vec![min_x, max_x, min_y, max_y],))?)
                }
                None => None,
            };

            // Call the JAX solve_mpc function
            let solve_mpc = mpc_module.getattr("solve_mpc")?;
            let result = solve_mpc.call1((
                initial_pos,
                initial_vel,
                target_pos,
                obstacles,
                field_bounds,
                vel_limit,
            ))?;

            // Extract the result - convert numpy array to Python list first
            let result_list = result.call_method0("tolist")?;
            let control_data: Vec<f64> = result_list.extract()?;

            // Restore stdout and capture any prints
            sys.setattr("stdout", original_stdout)?;
            let captured_output: String = stdout_capture.call_method0("getvalue")?.extract()?;

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

            if control_data.len() >= 2 {
                Ok(Vector2::new(control_data[0], control_data[1]))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid control output size"))
            }
        })
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
