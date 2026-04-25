//! iLQR solver for the translational MPC problem.
//!
//! Standard iLQR: backward Riccati pass produces feedforward `k` and feedback
//! `K` gains; forward rollout applies `u = u_ref + α·k + K·(x − x_ref)` with a
//! short Armijo line search over `α`. Levenberg-style regularisation is
//! applied to `Q_uu` and grown on rejected steps / non-PD matrices.
//!
//! `solve` runs two initialisations — warm-started rollout (if provided) and a
//! constant-velocity straight line aimed at the target — and returns whichever
//! converges to lower cost.

use nalgebra::{Matrix2, Matrix2x4, Matrix4, Vector2, Vector4};

use crate::cost::{stage_cost_scalar, stage_derivs, terminal_cost_scalar, terminal_derivs};
use crate::dynamics::{step, step_with_jacobians};
use crate::types::{
    Control, MpcTarget, RobotParams, RobotState, SolveResult, SolverConfig, State, Trajectory,
    Vec2,
};

/// Forward rollout of a control sequence, returning the trajectory and total cost.
fn rollout(
    x0: &State,
    controls: &[Control],
    u_prev_first: &Control,
    heading_traj: &[f64],
    target: &MpcTarget,
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
        cost += stage_cost_scalar(&states[k], &u_k, &u_prev, target);
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
        let sd = stage_derivs(&states[k], &controls[k], &u_prev, target);
        let h_k = heading_traj.get(k).copied().unwrap_or(0.0);
        let (_, fx, fu) = step_with_jacobians(&states[k], &controls[k], h_k, dt, params);

        let q_x: Vector4<f64> = sd.lx + fx.transpose() * v_x;
        let q_u: Vector2<f64> = sd.lu + fu.transpose() * v_x;
        let q_xx: Matrix4<f64> = sd.lxx + fx.transpose() * v_xx * fx;
        let q_uu_raw: Matrix2<f64> = sd.luu + fu.transpose() * v_xx * fu;
        let q_uu: Matrix2<f64> = q_uu_raw + reg * Matrix2::<f64>::identity();
        let q_ux: Matrix2x4<f64> = sd.lux + fu.transpose() * v_xx * fx;

        let q_uu_sym = 0.5 * (q_uu + q_uu.transpose());
        let q_uu_inv = q_uu_sym.try_inverse()?;

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
        cost += stage_cost_scalar(&new_states[k], &u_new, &u_prev, target);
        let x_next = step(&new_states[k], &u_new, h_k, dt, params);
        new_states.push(x_next);
        u_prev = u_new;
    }
    cost += terminal_cost_scalar(&new_states[n], target);
    (new_states, new_controls, cost)
}

fn run_ilqr(
    state: &RobotState,
    u_init: &[Control],
    u_prev_first: &Control,
    heading_traj: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
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
                params,
                cfg.dt,
            );
            let expected = alpha * back.expected_dv1 + alpha * alpha * back.expected_dv2;
            let actual = new_cost - cost;
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

/// Multi-start iLQR: warm-start (if provided) and straight-line-to-target.
/// Returns whichever converges to lower cost.
pub fn solve(
    state: RobotState,
    heading_traj: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
    warm_start: Option<&Trajectory>,
    cfg: &SolverConfig,
) -> SolveResult {
    let n = cfg.horizon;
    let u_prev_first = warm_start
        .and_then(|t| t.controls.first().copied())
        .unwrap_or_else(Vector2::zeros);

    // Init 1: warm-start (shifted by one stage), padded with the last commanded
    // value. Falls back to zeros if no warm start available.
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

    // Init 2: constant velocity that would reach the target in `horizon · dt`.
    // Magnitude clipped so we don't start from a wildly large iterate.
    let horizon_time = cfg.dt * n as f64;
    let dir = target.p - state.pos;
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
        cfg,
    );
    let r_line = run_ilqr(
        &state,
        &straight_controls,
        &u_prev_first,
        heading_traj,
        target,
        params,
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

    fn params() -> RobotParams {
        RobotParams::default_hand_tuned()
    }

    fn cfg_long() -> SolverConfig {
        // Use a longer horizon for tests so the solver has time to slow down.
        SolverConfig {
            horizon: 50,
            ..SolverConfig::default()
        }
    }

    #[test]
    fn at_rest_at_target_emits_zero_control() {
        // The whole point of the `||u||²` term: at the target with zero state
        // perturbation, iLQR must produce ~zero commanded velocity. This is
        // the unit-test version of "no +500 oscillation at target".
        let cfg = cfg_long();
        let target = MpcTarget::goto(Vec2::new(1000.0, 500.0));
        let state = RobotState {
            pos: target.p,
            vel: Vec2::zeros(),
        };
        let headings = vec![0.0; cfg.horizon + 1];
        let r = solve(state, &headings, &target, &params(), None, &cfg);
        let u0 = r.trajectory.controls[0];
        assert!(
            u0.norm() < 1.0,
            "control at target should be ~0 mm/s, got {}",
            u0.norm()
        );
    }

    #[test]
    fn converges_to_static_target() {
        let cfg = cfg_long();
        let target = MpcTarget::goto(Vec2::new(2000.0, 0.0));
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let headings = vec![0.0; cfg.horizon + 1];
        let r = solve(state, &headings, &target, &params(), None, &cfg);
        // Final state in the planned trajectory should be near the target with
        // small terminal velocity.
        let last = r.trajectory.states.last().unwrap();
        let p_err = ((last[0] - target.p.x).powi(2) + (last[1] - target.p.y).powi(2)).sqrt();
        let v_err = (last[2] * last[2] + last[3] * last[3]).sqrt();
        assert!(p_err < 200.0, "terminal position error {} mm", p_err);
        assert!(v_err < 200.0, "terminal velocity {} mm/s", v_err);
    }

    #[test]
    fn warm_start_does_not_regress() {
        let cfg = SolverConfig::default();
        let target = MpcTarget::goto(Vec2::new(1500.0, 0.0));
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        let headings = vec![0.0; cfg.horizon + 1];
        let cold = solve(state, &headings, &target, &params(), None, &cfg);
        let warm = solve(
            state,
            &headings,
            &target,
            &params(),
            Some(&cold.trajectory),
            &cfg,
        );
        // Warm start should at least not be worse than cold by a meaningful margin.
        assert!(
            warm.final_cost <= cold.final_cost + 1.0,
            "warm cost {} should not exceed cold cost {} by much",
            warm.final_cost,
            cold.final_cost
        );
    }
}
