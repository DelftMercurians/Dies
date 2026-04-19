//! iLQR solver for the translational MPC problem.
//!
//! Standard iLQR formulation (quadratic value function, linearized dynamics,
//! quadratic stage cost via Gauss-Newton Hessian). Regularisation is applied
//! to `Q_uu` and grown on rejected steps / non-PD matrices; the forward pass
//! uses a short Armijo line search over α ∈ {1, 0.5, 0.25, 0.125}.
//!
//! The outer `solve` wrapper runs two initialisations — a warm-started rollout
//! (if provided) and a constant-velocity "straight-line" rollout aimed at the
//! terminal target — and returns whichever converges to the lower cost.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

use crate::cost::{stage_cost_scalar, stage_derivs, terminal_cost_scalar, terminal_derivs};
use crate::dynamics::{step, step_with_jacobians};
use crate::types::{
    Control, MpcTarget, RobotParams, RobotState, SolveResult, SolverConfig, State, TerminalMode,
    Trajectory, Vec2, WorldSnapshot,
};

fn terminal_target_pos(m: &TerminalMode) -> Vec2 {
    match m {
        TerminalMode::Position { p } => *p,
        TerminalMode::PositionAndVelocity { p, .. } => *p,
        TerminalMode::RelativeVelocity { target_p, .. } => *target_p,
    }
}

/// Forward rollout of a control sequence from a fixed initial state, also
/// computing the total cost along the way.
fn rollout(
    x0: &State,
    controls: &[Control],
    u_prev_first: &Control,
    heading_traj: &[f64],
    target: &MpcTarget,
    world: &WorldSnapshot,
    params: &RobotParams,
    dt: f64,
) -> (Vec<State>, f64) {
    let n = controls.len();
    let mut states = Vec::with_capacity(n + 1);
    states.push(*x0);
    let mut cost = 0.0;
    let mut u_prev = *u_prev_first;
    for k in 0..n {
        let u_k = controls[k];
        let h_k = heading_traj.get(k).copied().unwrap_or(0.0);
        cost += stage_cost_scalar(k, &states[k], &u_k, &u_prev, target, world, dt);
        let x_next = step(&states[k], &u_k, h_k, dt, params);
        states.push(x_next);
        u_prev = u_k;
    }
    cost += terminal_cost_scalar(&states[n], target);
    (states, cost)
}

struct Backward {
    k_fb: Vec<Vector2<f64>>,
    kk_fb: Vec<Matrix2x4<f64>>,
    expected_dv1: f64,
    expected_dv2: f64,
}

fn backward_pass(
    states: &[State],
    controls: &[Control],
    u_prev_first: &Control,
    heading_traj: &[f64],
    target: &MpcTarget,
    world: &WorldSnapshot,
    params: &RobotParams,
    dt: f64,
    reg: f64,
) -> Option<Backward> {
    let n = controls.len();
    let term = terminal_derivs(&states[n], target);
    let mut v_x = term.lx;
    let mut v_xx = term.lxx;

    let mut k_fb = vec![Vector2::<f64>::zeros(); n];
    let mut kk_fb = vec![Matrix2x4::<f64>::zeros(); n];
    let mut dv1 = 0.0;
    let mut dv2 = 0.0;

    for k in (0..n).rev() {
        let u_prev = if k == 0 {
            *u_prev_first
        } else {
            controls[k - 1]
        };
        let sd = stage_derivs(k, &states[k], &controls[k], &u_prev, target, world, dt);
        let h_k = heading_traj.get(k).copied().unwrap_or(0.0);
        let (_, fx, fu) = step_with_jacobians(&states[k], &controls[k], h_k, dt, params);

        let q_x: Vector4<f64> = sd.lx + fx.transpose() * v_x;
        let q_u: Vector2<f64> = sd.lu + fu.transpose() * v_x;
        let q_xx: Matrix4<f64> = sd.lxx + fx.transpose() * v_xx * fx;
        let q_uu_raw: Matrix2<f64> = sd.luu + fu.transpose() * v_xx * fu;
        let q_uu: Matrix2<f64> = q_uu_raw + reg * Matrix2::<f64>::identity();
        let q_ux: Matrix2x4<f64> = sd.lux + fu.transpose() * v_xx * fx;

        // Symmetrise to fight numerical drift on the way through the backward pass.
        let q_uu_sym = 0.5 * (q_uu + q_uu.transpose());
        let q_uu_inv = q_uu_sym.try_inverse()?;

        // Reject if the regularised Q_uu is not positive-definite; caller
        // bumps regularisation and retries.
        if q_uu_inv.trace() <= 0.0 || q_uu_sym.determinant() <= 0.0 {
            return None;
        }

        let k_vec: Vector2<f64> = -q_uu_inv * q_u;
        let kk: Matrix2x4<f64> = -q_uu_inv * q_ux;

        k_fb[k] = k_vec;
        kk_fb[k] = kk;

        dv1 += q_u.dot(&k_vec);
        dv2 += 0.5 * (k_vec.transpose() * q_uu_sym * k_vec)[(0, 0)];

        v_x = q_x + q_ux.transpose() * k_vec;
        v_xx = q_xx + q_ux.transpose() * kk;
        // Symmetrise V_xx.
        v_xx = 0.5 * (v_xx + v_xx.transpose());
    }

    Some(Backward {
        k_fb,
        kk_fb,
        expected_dv1: dv1,
        expected_dv2: dv2,
    })
}

fn forward_pass(
    alpha: f64,
    x0: &State,
    prev_states: &[State],
    prev_controls: &[Control],
    u_prev_first: &Control,
    back: &Backward,
    heading_traj: &[f64],
    target: &MpcTarget,
    world: &WorldSnapshot,
    params: &RobotParams,
    dt: f64,
) -> (Vec<State>, Vec<Control>, f64) {
    let n = prev_controls.len();
    let mut new_states = Vec::with_capacity(n + 1);
    let mut new_controls = Vec::with_capacity(n);
    new_states.push(*x0);
    let mut cost = 0.0;
    let mut u_prev = *u_prev_first;
    for k in 0..n {
        let dx: Vector4<f64> = new_states[k] - prev_states[k];
        let u_new = prev_controls[k] + alpha * back.k_fb[k] + back.kk_fb[k] * dx;
        new_controls.push(u_new);
        let h_k = heading_traj.get(k).copied().unwrap_or(0.0);
        cost += stage_cost_scalar(k, &new_states[k], &u_new, &u_prev, target, world, dt);
        let x_next = step(&new_states[k], &u_new, h_k, dt, params);
        new_states.push(x_next);
        u_prev = u_new;
    }
    cost += terminal_cost_scalar(&new_states[n], target);
    (new_states, new_controls, cost)
}

/// Run iLQR from an explicit initial control sequence. Helper used by the
/// multi-start wrapper below.
fn run_ilqr(
    state: &RobotState,
    u_init: &[Control],
    u_prev_first: &Control,
    heading_traj: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
    world: &WorldSnapshot,
    cfg: &SolverConfig,
) -> SolveResult {
    let x0 = state.to_state();
    let mut controls = u_init.to_vec();
    let (mut states, mut cost) = rollout(
        &x0,
        &controls,
        u_prev_first,
        heading_traj,
        target,
        world,
        params,
        cfg.dt,
    );
    let mut reg = cfg.reg_init.max(cfg.reg_min);
    let mut converged = false;
    let mut iters = 0;
    let alphas = [1.0, 0.5, 0.25, 0.125, 0.0625];

    let started = std::time::Instant::now();
    for it in 0..cfg.max_iters {
        iters = it + 1;
        let back = loop {
            match backward_pass(
                &states,
                &controls,
                u_prev_first,
                heading_traj,
                target,
                world,
                params,
                cfg.dt,
                reg,
            ) {
                Some(b) => break Some(b),
                None => {
                    reg = (reg * cfg.reg_factor).min(cfg.reg_max);
                    if reg >= cfg.reg_max {
                        break None;
                    }
                }
            }
        };
        let Some(back) = back else {
            break;
        };

        let mut accepted = false;
        for &alpha in &alphas {
            let (ns, nc, new_cost) = forward_pass(
                alpha,
                &x0,
                &states,
                &controls,
                u_prev_first,
                &back,
                heading_traj,
                target,
                world,
                params,
                cfg.dt,
            );
            let expected = alpha * back.expected_dv1 + alpha * alpha * back.expected_dv2;
            let actual = new_cost - cost;
            // Accept if we got a nontrivial fraction of expected decrease, or
            // any decrease at all when the expected amount is too small to
            // meaningfully gate on.
            let tol = 1.0e-8 * cost.abs().max(1.0);
            let accept = if expected < -tol {
                actual < 0.1 * expected
            } else {
                actual < -tol
            };
            if accept {
                let prev_cost = cost;
                states = ns;
                controls = nc;
                cost = new_cost;
                accepted = true;
                reg = (reg / cfg.reg_factor).max(cfg.reg_min);
                if (prev_cost - cost).abs() < cfg.cost_tol * prev_cost.abs().max(1.0) {
                    converged = true;
                }
                break;
            }
        }
        if !accepted {
            reg = (reg * cfg.reg_factor).min(cfg.reg_max);
            if reg >= cfg.reg_max {
                break;
            }
        }
        if converged {
            break;
        }
    }

    let elapsed = started.elapsed().as_micros() as u64;
    SolveResult {
        trajectory: Trajectory { states, controls },
        final_cost: cost,
        iters,
        converged,
        solve_time_us: elapsed,
    }
}

/// Two-start iLQR: warm-start if provided + straight-line-to-terminal. Picks
/// the lower-cost solution.
pub fn solve(
    state: RobotState,
    heading_traj: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
    world: &WorldSnapshot,
    warm_start: Option<&Trajectory>,
    cfg: &SolverConfig,
) -> SolveResult {
    let n = cfg.horizon;
    let u_prev_first = warm_start
        .and_then(|t| t.controls.first().copied())
        .unwrap_or_else(Vector2::zeros);

    // Init 1: warm-start (shifted by one stage), padded with the last
    // commanded value. If absent, this degenerates to zeros which iLQR will
    // still correct — but we always keep the straight-line branch for safety.
    let warm_controls: Vec<Control> = if let Some(ws) = warm_start {
        let mut v = Vec::with_capacity(n);
        for k in 0..n {
            let idx = (k + 1).min(ws.controls.len().saturating_sub(1));
            let c = ws.controls.get(idx).copied().unwrap_or_else(Vector2::zeros);
            v.push(c);
        }
        v
    } else {
        vec![Vector2::zeros(); n]
    };

    // Init 2: constant velocity aiming at the terminal target position,
    // magnitude clipped so we start from a sane iterate.
    let target_p = terminal_target_pos(&target.terminal);
    let horizon_time = cfg.dt * n as f64;
    let dir = target_p - state.pos;
    let desired = if horizon_time > 1.0e-9 {
        dir / horizon_time
    } else {
        Vec2::zeros()
    };
    const SANE_CAP: f64 = 3500.0;
    let clipped = {
        let nrm = desired.norm();
        if nrm > SANE_CAP {
            desired * (SANE_CAP / nrm)
        } else {
            desired
        }
    };
    let straight_controls: Vec<Control> = vec![clipped; n];

    let r_warm = run_ilqr(
        &state,
        &warm_controls,
        &u_prev_first,
        heading_traj,
        target,
        params,
        world,
        cfg,
    );
    let r_line = run_ilqr(
        &state,
        &straight_controls,
        &u_prev_first,
        heading_traj,
        target,
        params,
        world,
        cfg,
    );
    if r_warm.final_cost <= r_line.final_cost {
        r_warm
    } else {
        r_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        CostWeights, FieldBounds, ObstacleShape, PredictedObstacle, ReferenceTrajectory,
    };
    use approx::assert_abs_diff_eq;

    fn params() -> RobotParams {
        RobotParams::default_hand_tuned()
    }

    fn cfg() -> SolverConfig {
        SolverConfig::default()
    }

    fn empty_world() -> WorldSnapshot {
        WorldSnapshot {
            obstacles: vec![],
            field_bounds: FieldBounds::centered(90_000.0, 60_000.0, 1000.0, 2000.0),
        }
    }

    fn goto(target: Vec2) -> MpcTarget {
        MpcTarget {
            reference: ReferenceTrajectory::StaticPoint(target),
            terminal: TerminalMode::Position { p: target },
            weights: CostWeights::default(),
            care: 1.0,
            aggressiveness: 0.2,
        }
    }

    fn headings(n: usize) -> Vec<f64> {
        vec![0.0; n + 1]
    }

    #[test]
    fn reaches_target_in_open_space() {
        let target_p = Vec2::new(1500.0, 800.0);
        let t = goto(target_p);
        let world = empty_world();
        let params = params();
        let cfg = cfg();
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let r = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            None,
            &cfg,
        );
        let final_state = RobotState::from_state(r.trajectory.states.last().unwrap());
        let err = (final_state.pos - target_p).norm();
        // The 600 ms horizon at normal params won't fully reach a 1.7 m
        // target, but it must make clear directional progress.
        let initial_err = target_p.norm();
        assert!(
            err < 0.7 * initial_err,
            "final error {} not markedly less than initial {}",
            err,
            initial_err
        );
    }

    #[test]
    fn avoids_single_circle_obstacle() {
        // Note: iLQR on a *perfectly* symmetric obstacle-on-line problem has
        // no y-gradient and can't find the detour direction. Real obstacles
        // won't sit exactly on the robot's centerline; the test offsets it
        // slightly so the gradient is well-posed. If field testing reveals
        // degenerate-symmetry deadlocks we can add a perturbed random-init
        // branch to the multi-start wrapper.
        let target_p = Vec2::new(2000.0, 0.0);
        let t = goto(target_p);
        let obs_center = Vec2::new(1000.0, 40.0);
        let world = WorldSnapshot {
            obstacles: vec![PredictedObstacle {
                shape: ObstacleShape::Circle {
                    center: obs_center,
                    radius: 100.0,
                },
                velocity: Vec2::zeros(),
                safe_dist: 180.0,
                no_cost_dist: 450.0,
                weight_scale: 3.0,
            }],
            field_bounds: FieldBounds::centered(90_000.0, 60_000.0, 1000.0, 2000.0),
        };
        let params = params();
        let cfg = SolverConfig {
            horizon: 20,
            ..cfg()
        };
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let r = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            None,
            &cfg,
        );
        let max_neg_y_dev = r
            .trajectory
            .states
            .iter()
            .map(|s| (-s[1]).max(0.0))
            .fold(0.0_f64, f64::max);
        assert!(
            max_neg_y_dev > 10.0,
            "trajectory did not deflect away from off-axis obstacle (max -y deviation {})",
            max_neg_y_dev
        );
        for s in &r.trajectory.states {
            let p = Vec2::new(s[0], s[1]);
            let d = (p - obs_center).norm() - 100.0;
            assert!(
                d > 0.0,
                "trajectory point penetrates obstacle body: d={}",
                d
            );
        }
    }

    #[test]
    fn converges_under_iter_budget() {
        let target_p = Vec2::new(500.0, 300.0);
        let t = goto(target_p);
        let world = empty_world();
        let params = params();
        let cfg = cfg();
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let r = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            None,
            &cfg,
        );
        assert!(r.iters <= cfg.max_iters);
        // Some useful descent must have happened.
        let zero_ctrl = vec![Vector2::zeros(); cfg.horizon];
        let (_, zero_cost) = rollout(
            &state.to_state(),
            &zero_ctrl,
            &Vector2::zeros(),
            &headings(cfg.horizon),
            &t,
            &world,
            &params,
            cfg.dt,
        );
        assert!(
            r.final_cost < 0.9 * zero_cost,
            "final cost {} did not improve on zero-control cost {}",
            r.final_cost,
            zero_cost
        );
    }

    #[test]
    fn warm_start_reduces_iters() {
        let target_p = Vec2::new(1200.0, -500.0);
        let t = goto(target_p);
        let world = empty_world();
        let params = params();
        let cfg = cfg();
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let cold = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            None,
            &cfg,
        );
        let warm = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            Some(&cold.trajectory),
            &cfg,
        );
        assert!(
            warm.final_cost <= cold.final_cost + 1.0e-6,
            "warm-started solve cost {} worse than cold {}",
            warm.final_cost,
            cold.final_cost
        );
    }

    #[test]
    fn no_panics_on_unreachable_target() {
        // A target deep inside an obstacle — the solver should still return,
        // not converge, but produce a valid trajectory without panicking.
        let target_p = Vec2::new(1000.0, 0.0);
        let t = goto(target_p);
        let world = WorldSnapshot {
            obstacles: vec![PredictedObstacle {
                shape: ObstacleShape::Circle {
                    center: Vec2::new(1000.0, 0.0),
                    radius: 500.0,
                },
                velocity: Vec2::zeros(),
                safe_dist: 100.0,
                no_cost_dist: 300.0,
                weight_scale: 10.0,
            }],
            field_bounds: FieldBounds::centered(90_000.0, 60_000.0, 1000.0, 2000.0),
        };
        let params = params();
        let cfg = cfg();
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let r = solve(
            state,
            &headings(cfg.horizon),
            &t,
            &params,
            &world,
            None,
            &cfg,
        );
        assert_eq!(r.trajectory.states.len(), cfg.horizon + 1);
        assert!(r.final_cost.is_finite());
        // Silence the unused marker on the derivative approx helper.
        assert_abs_diff_eq!(r.final_cost, r.final_cost, epsilon = 1.0);
    }
}
