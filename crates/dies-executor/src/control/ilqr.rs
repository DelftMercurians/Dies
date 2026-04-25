//! Thin integration wrapper around `dies-mpc`'s iLQR solver.
//!
//! Owns per-robot warm-start trajectories plus the global `RobotParams` and
//! `SolverConfig`. Translates the executor's `PlayerControlInput` + `TeamData`
//! into `dies-mpc` types, calls the solver once per controllable robot, and
//! returns a `HashMap<PlayerId, Vector2>` of velocity overrides that the team
//! controller applies *after* the per-controller MTP update (so iLQR wins when
//! enabled).
//!
//! Robots with no position target are omitted from the returned map — callers
//! fall through to the existing velocity-passthrough path.
//!
//! Obstacle and field-boundary avoidance are NOT handled here. The minimal
//! `dies-mpc` is pure dynamics + tracking cost; obstacle behaviour belongs
//! above this layer (e.g. via the MTP path), or in a future iteration once
//! the simpler stack is validated.

use std::{collections::HashMap, fs, path::Path};

use dies_core::{DebugColor, PlayerId, TeamData, Vector2};
use dies_mpc::{self, MpcTarget, RobotParams, RobotState, SolverConfig, Trajectory};

use super::{player_controller::PlayerController, player_input::PlayerControlInput};

/// Owner of iLQR state. One instance per `TeamController`.
pub struct IlqrController {
    params: RobotParams,
    cfg: SolverConfig,
    warm_starts: HashMap<PlayerId, Trajectory>,
}

impl IlqrController {
    pub fn new() -> Self {
        Self::with_params(RobotParams::default_hand_tuned())
    }

    pub fn with_params(params: RobotParams) -> Self {
        Self {
            params,
            cfg: SolverConfig::default(),
            warm_starts: HashMap::new(),
        }
    }

    /// Load `RobotParams` from a JSON file. On any read/parse error, falls back
    /// to the hand-tuned defaults (and re-seeds the file when missing).
    pub fn load_or_insert(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match fs::read_to_string(path) {
            Ok(contents) => match serde_json::from_str::<RobotParams>(&contents) {
                Ok(params) => {
                    tracing::info!("Loaded iLQR RobotParams from {}", path.display());
                    Self::with_params(params)
                }
                Err(err) => {
                    tracing::warn!(
                        "Failed to parse iLQR params at {}: {} — using defaults",
                        path.display(),
                        err
                    );
                    Self::new()
                }
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                let params = RobotParams::default_hand_tuned();
                if let Err(err) = fs::write(path, serde_json::to_string_pretty(&params).unwrap()) {
                    tracing::warn!(
                        "Failed to seed iLQR params file {}: {}",
                        path.display(),
                        err
                    );
                }
                Self::with_params(params)
            }
            Err(err) => {
                tracing::warn!(
                    "Failed to read iLQR params at {}: {} — using defaults",
                    path.display(),
                    err
                );
                Self::new()
            }
        }
    }

    /// Solve iLQR for every controllable robot with a position target.
    pub fn compute_batch_control(
        &mut self,
        controllers: &HashMap<PlayerId, PlayerController>,
        inputs: &HashMap<PlayerId, PlayerControlInput>,
        world: &TeamData,
    ) -> HashMap<PlayerId, Vector2> {
        let mut out = HashMap::new();

        // Drop warm-starts for players we no longer control so we don't leak
        // memory across substitutions.
        self.warm_starts
            .retain(|id, _| controllers.contains_key(id));

        for (id, controller) in controllers.iter() {
            let Some(input) = inputs.get(id) else {
                continue;
            };
            let Some(target_p) = input.position else {
                continue;
            };
            let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) else {
                continue;
            };

            let target = MpcTarget::goto(target_p);
            let heading_traj = vec![player_data.yaw.radians(); self.cfg.horizon + 1];
            let state = RobotState {
                pos: player_data.position,
                vel: player_data.velocity,
            };

            let result = dies_mpc::solve(
                state,
                &heading_traj,
                &target,
                &self.params,
                self.warm_starts.get(id),
                &self.cfg,
            );

            let mut cmd = result
                .trajectory
                .controls
                .first()
                .copied()
                .unwrap_or_else(Vector2::zeros);

            // Clip to configured max speed so iLQR can't command something the
            // basestation will saturate anyway.
            let max_speed = input.speed_limit.unwrap_or(controller.get_max_speed());
            let nrm = cmd.norm();
            if nrm > max_speed {
                cmd *= max_speed / nrm;
            }

            // Render the planned trajectory as debug line segments so the field
            // canvas can show it.
            for (i, window) in result.trajectory.states.windows(2).enumerate() {
                dies_core::debug_line(
                    format!("p{}.ilqr.trajectory.seg{:02}", id, i),
                    window[0].xy(),
                    window[1].xy(),
                    DebugColor::Purple,
                );
            }

            self.warm_starts.insert(*id, result.trajectory);
            out.insert(*id, cmd);

            dies_core::debug_value(
                format!("p{}.ilqr.solve_us", id),
                result.solve_time_us as f64,
            );
            dies_core::debug_value(format!("p{}.ilqr.iters", id), result.iters as f64);
            dies_core::debug_value(format!("p{}.ilqr.cost", id), result.final_cost);
        }
        out
    }
}
