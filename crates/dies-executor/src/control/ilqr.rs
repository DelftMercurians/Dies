//! Thin integration wrapper around `dies-mpc`'s iLQR solver.
//!
//! Owns per-robot warm-start trajectories plus the (currently global,
//! hand-tuned) `RobotParams` and `SolverConfig`. Translates the executor's
//! `PlayerControlInput` + `TeamData` into `dies-mpc` types, calls the solver
//! once per controllable robot, and returns a `HashMap<PlayerId, Vector2>`
//! of velocity overrides that the team controller applies *after* the
//! per-controller MTP update (so iLQR wins when enabled).
//!
//! Robots with no position target are omitted from the returned map —
//! callers fall through to the existing velocity-passthrough path.

use std::{collections::HashMap, fs, path::Path};

use dies_core::{
    Angle, BallData, DebugColor, FieldGeometry, GameState, PlayerData, PlayerId, TeamData,
    Vector2, BALL_RADIUS, PLAYER_RADIUS,
};
use dies_mpc::{
    self, CostWeights, FieldBounds, MpcTarget, ObstacleShape, PredictedObstacle,
    ReferenceTrajectory, RobotParams, RobotState, SolverConfig, TerminalMode, Trajectory,
    WorldSnapshot,
};

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

    /// Load `RobotParams` from a JSON file. On success, returns a controller
    /// using those params. On parse error or any IO error other than "not
    /// found", logs and falls back to hand-tuned defaults. If the file is
    /// missing, writes the current defaults to that path so it's easy to
    /// edit by hand or overwrite from sysid.
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
    /// Robots without a position target are absent from the returned map.
    pub fn compute_batch_control(
        &mut self,
        controllers: &HashMap<PlayerId, PlayerController>,
        inputs: &HashMap<PlayerId, PlayerControlInput>,
        world: &TeamData,
        avoid_goal_area_flags: &HashMap<PlayerId, bool>,
    ) -> HashMap<PlayerId, Vector2> {
        let mut out = HashMap::new();
        let Some(field) = world.field_geom.as_ref() else {
            return out;
        };
        let field_bounds = field_bounds_from(field);

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

            let avoid_goal_area = avoid_goal_area_flags.get(id).copied().unwrap_or(true);
            let obstacles = build_obstacles(*id, input, world, field, avoid_goal_area);
            let snapshot = WorldSnapshot {
                obstacles,
                field_bounds: field_bounds.clone(),
            };

            let target = MpcTarget {
                reference: ReferenceTrajectory::StaticPoint(target_p),
                terminal: TerminalMode::PositionAndVelocity {
                    p: target_p,
                    v: Vector2::zeros(),
                },
                weights: CostWeights::default(),
                care: input.care,
                aggressiveness: input.aggressiveness,
            };

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
                &snapshot,
                self.warm_starts.get(id),
                &self.cfg,
            );

            let mut cmd = result
                .trajectory
                .controls
                .first()
                .copied()
                .unwrap_or_else(Vector2::zeros);

            // Clip to the configured max speed so iLQR can't command something
            // the basestation will saturate anyway.
            let max_speed = input.speed_limit.unwrap_or(controller.get_max_speed());
            let nrm = cmd.norm();
            if nrm > max_speed {
                cmd *= max_speed / nrm;
            }

            // Publish the predicted trajectory as a chain of debug lines so
            // the webui field canvas can render it. World-frame coordinates,
            // matching the initial state we fed in.
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

fn field_bounds_from(field: &FieldGeometry) -> FieldBounds {
    FieldBounds::centered(
        field.field_length,
        field.field_width,
        field.penalty_area_depth,
        field.penalty_area_width,
    )
}

/// Translate the executor's world view into a list of `PredictedObstacle`s
/// for one robot. Mirrors the conventions used by
/// `team_controller::update` / `two_step_mtp` today.
fn build_obstacles(
    own_id: PlayerId,
    input: &PlayerControlInput,
    world: &TeamData,
    field: &FieldGeometry,
    avoid_goal_area: bool,
) -> Vec<PredictedObstacle> {
    let mut obstacles = Vec::with_capacity(16);

    // Ball obstacle — distance measured surface-to-robot-center.
    if should_avoid_ball(input, world) {
        if let Some(ball) = world.ball.as_ref() {
            obstacles.push(ball_obstacle(input, ball, world));
        }
    }

    // Ball-replacement corridor: penalise straying onto the ball-placement line.
    if let GameState::BallReplacement(target) = world.current_game_state.game_state {
        if let Some(ball) = world.ball.as_ref() {
            obstacles.push(PredictedObstacle {
                shape: ObstacleShape::Line {
                    start: ball.position.xy(),
                    end: target,
                },
                velocity: Vector2::zeros(),
                safe_dist: 500.0,
                no_cost_dist: 700.0,
                weight_scale: 2.0,
            });
        }
    }

    // Opponent robots.
    for opp in world.opp_players.iter() {
        obstacles.push(robot_obstacle(opp));
    }

    // Own teammates (excluding self). Matches MTP `avoid_opp_robots` flag —
    // naming is historical; the flag gates opponent avoidance. Teammates are
    // always avoided.
    for own in world.own_players.iter().filter(|p| p.id != own_id) {
        obstacles.push(robot_obstacle(own));
    }

    // Goal areas as rectangles when this robot is supposed to stay out of them.
    if avoid_goal_area {
        let hx = field.field_length * 0.5;
        let hy = field.penalty_area_width * 0.5;
        // Our goal area (negative-x end of the field).
        obstacles.push(PredictedObstacle {
            shape: ObstacleShape::Rectangle {
                min: Vector2::new(-hx - 500.0, -hy),
                max: Vector2::new(-hx + field.penalty_area_depth, hy),
            },
            velocity: Vector2::zeros(),
            safe_dist: 50.0,
            no_cost_dist: 250.0,
            weight_scale: 5.0,
        });
        // Opponent goal area (positive-x end).
        obstacles.push(PredictedObstacle {
            shape: ObstacleShape::Rectangle {
                min: Vector2::new(hx - field.penalty_area_depth, -hy),
                max: Vector2::new(hx + 500.0, hy),
            },
            velocity: Vector2::zeros(),
            safe_dist: 50.0,
            no_cost_dist: 250.0,
            weight_scale: 5.0,
        });
    }

    obstacles
}

fn should_avoid_ball(input: &PlayerControlInput, world: &TeamData) -> bool {
    input.avoid_ball
        || matches!(
            world.current_game_state.game_state,
            GameState::PreparePenalty | GameState::Stop | GameState::BallReplacement(_)
        )
}

fn ball_obstacle(
    input: &PlayerControlInput,
    ball: &BallData,
    world: &TeamData,
) -> PredictedObstacle {
    // In `Stop`, SSL rules mandate a much wider ball exclusion zone — match
    // the 800 mm value used by TwoStepMTP.
    let stop_mode = matches!(world.current_game_state.game_state, GameState::Stop);
    let safe_dist = if stop_mode {
        800.0
    } else {
        PLAYER_RADIUS * 1.05 + 50.0 * (input.care + input.avoid_ball_care)
    };
    let no_cost_dist = safe_dist + 200.0;
    PredictedObstacle {
        shape: ObstacleShape::Circle {
            center: ball.position.xy(),
            radius: BALL_RADIUS,
        },
        velocity: ball.velocity.xy(),
        safe_dist,
        no_cost_dist,
        weight_scale: 1.0,
    }
}

fn robot_obstacle(p: &PlayerData) -> PredictedObstacle {
    PredictedObstacle {
        shape: ObstacleShape::Circle {
            center: p.position,
            radius: PLAYER_RADIUS,
        },
        velocity: p.velocity,
        // Matches `two_step_mtp::robot_scare` (PLAYER_RADIUS · 1.05 + small
        // extra margin) — robot surface to own-center.
        safe_dist: PLAYER_RADIUS * 1.05,
        no_cost_dist: PLAYER_RADIUS * 2.2,
        weight_scale: 1.0,
    }
}

// Silence unused imports in case Angle only shows up via PlayerData's field.
const _: fn(&Angle) = |_| {};
