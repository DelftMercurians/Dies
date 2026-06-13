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
//! Obstacle avoidance is handled by building a per-robot soft-obstacle set
//! (`build_obstacles`) in team-relative coordinates and attaching it to the
//! `MpcTarget`: other robots as constant-velocity keep-out disks, the field
//! walls as keep-in half-planes, and the defense areas as keep-out boxes. The
//! barriers themselves (clearance, gradient, Gauss-Newton Hessian) live in
//! `dies_mpc::obstacle`; geometry and tuning come from `ObstacleConfig`.

use std::collections::HashMap;

use dies_core::{DebugColor, PlayerId, TeamData, Vector2, PLAYER_RADIUS};
use dies_mpc::{
    self, Control, MpcTarget, Obstacle, ObstacleConfig, ObstacleShape, RobotParams, RobotState,
    SolverConfig, Trajectory,
};
use nalgebra::Matrix2;

use super::{player_controller::PlayerController, player_input::PlayerControlInput};

/// iLQR output for one robot: the immediate velocity command and the heading
/// setpoint to forward to the onboard yaw controller. Both global frame.
#[derive(Clone, Copy, Debug)]
pub struct IlqrCommand {
    pub velocity: Vector2,
    pub heading: f64,
}

/// Owner of iLQR state. One instance per `TeamController`.
pub struct IlqrController {
    params: RobotParams,
    cfg: SolverConfig,
    warm_starts: HashMap<PlayerId, Trajectory>,
    /// Previously *commanded* setpoint per player, global frame. Used for the
    /// hard per-axis acceleration cap on the solver output (mirrors the
    /// `last_vel` pattern in `two_step_mtp`: rate-limit the setpoint, not the
    /// measured velocity).
    last_cmd: HashMap<PlayerId, Vector2>,
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
            last_cmd: HashMap::new(),
        }
    }

    /// Replace the tuning parameters live (from a settings update). Warm-starts
    /// are kept — a slightly stale seed is still a fine initial guess and the
    /// solver re-converges within a tick.
    pub fn set_params(&mut self, params: RobotParams) {
        self.params = params;
    }

    /// Build the soft obstacle set for one robot, in team-relative coordinates:
    /// every other robot as a constant-velocity keep-out disk (gated by
    /// `avoid_robots`), the field walls as keep-in half-planes, and the two
    /// defense areas as keep-out boxes (gated by `avoid_defense_area`).
    fn build_obstacles(
        &self,
        id: PlayerId,
        input: &PlayerControlInput,
        world: &TeamData,
    ) -> Vec<Obstacle> {
        let cfg: &ObstacleConfig = &self.params.obstacles;
        let mut obstacles = Vec::new();

        // Other robots → constant-velocity keep-out disks. Own and opponent ids
        // share the same numbering, so the self-skip filters own players only.
        if input.avoid_robots {
            let keepout_r = 2.0 * PLAYER_RADIUS + cfg.robot_clearance;
            for other in world
                .own_players
                .iter()
                .filter(|p| p.id != id)
                .chain(world.opp_players.iter())
            {
                obstacles.push(Obstacle {
                    shape: ObstacleShape::Circle {
                        center: other.position,
                        radius: keepout_r,
                    },
                    vel: other.velocity,
                    vel_cap_t: cfg.robot_extrapolation,
                    weight: cfg.weight,
                    influence: cfg.influence,
                });
            }
        }

        if let Some(field) = world.field_geom.as_ref() {
            // Field walls → keep-in half-planes (clearance = offset − normal·p),
            // inset from the physical boundary by `wall_margin`.
            let x_max = field.field_length / 2.0 + field.boundary_width - cfg.wall_margin;
            let y_max = field.field_width / 2.0 + field.boundary_width - cfg.wall_margin;
            for (normal, offset) in [
                (Vector2::new(1.0, 0.0), x_max),
                (Vector2::new(-1.0, 0.0), x_max),
                (Vector2::new(0.0, 1.0), y_max),
                (Vector2::new(0.0, -1.0), y_max),
            ] {
                obstacles.push(Obstacle::fixed(
                    ObstacleShape::HalfPlane { normal, offset },
                    cfg.weight,
                    cfg.influence,
                ));
            }

            // Defense areas → keep-out boxes, extended behind the goal line so a
            // robot near one is always pushed back out toward the field.
            if input.avoid_defense_area {
                let hl = field.field_length / 2.0;
                let depth = field.penalty_area_depth;
                let half_w = field.penalty_area_width / 2.0;
                let m = cfg.defense_margin;
                let back = field.boundary_width + m;
                // Own defense area (−x) and opponent defense area (+x).
                for (min, max) in [
                    (
                        Vector2::new(-hl - back, -half_w - m),
                        Vector2::new(-hl + depth + m, half_w + m),
                    ),
                    (
                        Vector2::new(hl - depth - m, -half_w - m),
                        Vector2::new(hl + back, half_w + m),
                    ),
                ] {
                    obstacles.push(Obstacle::fixed(
                        ObstacleShape::Box { min, max },
                        cfg.weight,
                        cfg.influence,
                    ));
                }
            }
        }

        obstacles
    }

    /// Solve iLQR for every controllable robot with a position target.
    pub fn compute_batch_control(
        &mut self,
        controllers: &HashMap<PlayerId, PlayerController>,
        inputs: &HashMap<PlayerId, PlayerControlInput>,
        world: &TeamData,
    ) -> HashMap<PlayerId, IlqrCommand> {
        let mut out = HashMap::new();

        // Drop warm-starts and last-command state for players we no longer
        // control so we don't leak memory across substitutions.
        self.warm_starts
            .retain(|id, _| controllers.contains_key(id));
        self.last_cmd.retain(|id, _| controllers.contains_key(id));

        // Loop dt from the latest vision frame. Drives both the solver's
        // integration step and the per-tick body-frame accel cap. Pinning the
        // solver's `dt` here is critical: with 50 Hz vision (~20 ms ticks) and
        // a hardcoded 60 ms solver dt, the planner's `controls[0]` is sized for
        // 60 ms but applied for 20 ms, producing systematic over-correction
        // and the visible decel→accel→decel lurching.
        let dt = world.dt.clamp(1.0e-3, 0.5);
        let cfg = SolverConfig {
            dt,
            ..self.cfg.clone()
        };

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

            let mut target = MpcTarget::goto(target_p);
            target.weights = self.params.weights.clone();
            // Heading: track the strategy's desired yaw when one is given;
            // otherwise leave heading free (zero weight) so the planner orients
            // the robot purely to optimise translation.
            match input.yaw {
                Some(yaw) => target.heading = yaw.radians(),
                None => {
                    target.heading = player_data.yaw.radians();
                    target.weights.heading = 0.0;
                }
            }
            target.obstacles = self.build_obstacles(*id, input, world);
            let state = RobotState {
                pos: player_data.position,
                vel: player_data.velocity,
                heading: player_data.yaw.radians(),
            };

            let result = dies_mpc::solve(
                state,
                &target,
                &self.params,
                self.warm_starts.get(id),
                &cfg,
            );

            // Control is `[vx_cmd, vy_cmd, theta_cmd]`: split into the velocity
            // override and the heading setpoint to forward to the yaw loop.
            let first_ctrl = result
                .trajectory
                .controls
                .first()
                .copied()
                .unwrap_or_else(Control::zeros);
            let mut cmd = Vector2::new(first_ctrl[0], first_ctrl[1]);
            let heading_setpoint = first_ctrl[2];

            // Clip to configured max speed so iLQR can't command something the
            // basestation will saturate anyway.
            let max_speed = input.speed_limit.unwrap_or(controller.get_max_speed());
            let min_speed = 10.0;
            let nrm = cmd.norm();
            if nrm > max_speed {
                cmd *= max_speed / nrm;
            }
            if nrm < min_speed {
                cmd *= 0.0;
            }

            // Hard per-axis acceleration cap on the *commanded setpoint* (not
            // measured velocity — same pattern as `two_step_mtp::last_vel`).
            // Body-frame so fwd/strafe limits map to motor reality.
            let last = self
                .last_cmd
                .get(id)
                .copied()
                .unwrap_or_else(Vector2::zeros);
            let yaw = player_data.yaw.radians();
            let (cy, sy) = (yaw.cos(), yaw.sin());
            let r = Matrix2::new(cy, -sy, sy, cy);
            let rt = r.transpose();
            let dv_global = cmd - last;
            let mut dv_body = rt * dv_global;
            for axis in 0..2 {
                let cap = self.params.accel_max[axis] * dt;
                dv_body[axis] = dv_body[axis].clamp(-cap, cap);
            }
            let new_cmd = last + r * dv_body;
            self.last_cmd.insert(*id, new_cmd);
            cmd = new_cmd;

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
            out.insert(
                *id,
                IlqrCommand {
                    velocity: cmd,
                    heading: heading_setpoint,
                },
            );

            dies_core::debug_value(
                format!("p{}.ilqr.solve_us", id),
                result.solve_time_us as f64,
            );
            dies_core::debug_value(format!("p{}.ilqr.iters", id), result.iters as f64);
            dies_core::debug_value(format!("p{}.ilqr.cost", id), result.final_cost);
            dies_core::debug_value(format!("p{}.ilqr.heading_sp", id), heading_setpoint);
        }
        out
    }
}
