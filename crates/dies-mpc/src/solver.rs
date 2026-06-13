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

use nalgebra::{Matrix3, Matrix3x5, Matrix5, Vector3, Vector5};

use crate::cost::{stage_cost_scalar, stage_derivs};
use crate::dynamics::{step, step_with_jacobians};
use crate::obstacle::{Obstacle, ObstacleShape};
use crate::types::{
    Control, MpcTarget, RobotParams, RobotState, SolveResult, SolverConfig, State, Trajectory, Vec2,
};

/// Largest sane seed speed [mm/s] — clips initial guesses so iLQR never starts
/// from a wildly large iterate.
const SANE_CAP: f64 = 3500.0;

fn clip_speed(v: Vec2) -> Vec2 {
    let nrm = v.norm();
    if nrm > SANE_CAP {
        v * (SANE_CAP / nrm)
    } else {
        v
    }
}

/// "Go around" control seeds for the head-on case.
///
/// A purely radial barrier has *no* lateral gradient when the robot drives
/// straight at an obstacle's centre (the start→obstacle→target collinear case),
/// so plain iLQR sits on that saddle and plain straight-line init plows through.
/// When a circular (robot) obstacle blocks the straight segment, this returns
/// two constant-velocity seeds — one bowing to each side via a waypoint clear of
/// the keep-out — so iLQR can descend into the left and right homotopy classes;
/// `solve` then keeps whichever has lower cost. Empty when nothing blocks.
fn detour_seeds(
    start: Vec2,
    target: Vec2,
    heading: f64,
    obstacles: &[Obstacle],
    n: usize,
    dt: f64,
) -> Vec<Vec<Control>> {
    let seg = target - start;
    let len = seg.norm();
    if len < 1.0e-6 || n == 0 {
        return Vec::new();
    }
    let dir = seg / len;
    let perp = Vec2::new(-dir.y, dir.x);

    // Most-blocking circular obstacle: one that projects onto the segment and
    // whose centre is nearest the line (smallest |perp|) within its reach.
    let mut best: Option<(f64, f64)> = None; // (along, clearance)
    let mut best_perp = f64::INFINITY;
    for ob in obstacles {
        let ObstacleShape::Circle { center, radius } = ob.shape else {
            continue;
        };
        let rel = center - start;
        let along = rel.dot(&dir);
        if along <= 0.0 || along >= len {
            continue; // not between start and target
        }
        let perp_dist = rel.dot(&perp).abs();
        let reach = radius + ob.influence;
        if perp_dist >= reach {
            continue; // straight path already clears it
        }
        if perp_dist < best_perp {
            best_perp = perp_dist;
            // Waypoint standoff: clear the keep-out and sit near the influence
            // edge so the refined trajectory settles just outside it.
            best = Some((along, radius + 0.5 * ob.influence + 80.0));
        }
    }

    let Some((along, clearance)) = best else {
        return Vec::new();
    };
    let proj = start + dir * along;
    let frac = (along / len).clamp(0.05, 0.95);
    let n1 = ((frac * n as f64).round() as usize).clamp(1, n.saturating_sub(1).max(1));
    let n2 = (n - n1).max(1);

    let mut seeds = Vec::with_capacity(2);
    for side in [1.0_f64, -1.0] {
        let waypoint = proj + perp * (side * clearance);
        let v1 = clip_speed((waypoint - start) / (n1 as f64 * dt));
        let v2 = clip_speed((target - waypoint) / (n2 as f64 * dt));
        let mut ctrl = Vec::with_capacity(n);
        for k in 0..n {
            let v = if k < n1 { v1 } else { v2 };
            ctrl.push(Control::new(v.x, v.y, heading));
        }
        seeds.push(ctrl);
    }
    seeds
}

/// Forward rollout of a control sequence, returning the trajectory and total cost.
fn rollout(
    x0: &State,
    controls: &[Control],
    u_prev_first: &Control,
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
        cost += stage_cost_scalar(&states[k], &u_k, &u_prev, target, k as f64 * dt);
        let x_next = step(&states[k], &u_k, dt, params);
        states.push(x_next);
        u_prev = u_k;
    }
    (states, cost)
}

struct Backward {
    k_fb: Vec<Vector3<f64>>,
    kk_fb: Vec<Matrix3x5<f64>>,
    expected_dv1: f64,
    expected_dv2: f64,
}

fn backward_pass(
    states: &[State],
    controls: &[Control],
    u_prev_first: &Control,
    target: &MpcTarget,
    params: &RobotParams,
    dt: f64,
    reg: f64,
) -> Option<Backward> {
    let n = controls.len();
    let mut v_x = Vector5::<f64>::zeros();
    let mut v_xx = Matrix5::<f64>::zeros();

    let mut k_fb = vec![Vector3::<f64>::zeros(); n];
    let mut kk_fb = vec![Matrix3x5::<f64>::zeros(); n];
    let mut dv1 = 0.0;
    let mut dv2 = 0.0;

    for k in (0..n).rev() {
        let u_prev = if k == 0 {
            *u_prev_first
        } else {
            controls[k - 1]
        };
        let sd = stage_derivs(&states[k], &controls[k], &u_prev, target, k as f64 * dt);
        let (_, fx, fu) = step_with_jacobians(&states[k], &controls[k], dt, params);

        let q_x: Vector5<f64> = sd.lx + fx.transpose() * v_x;
        let q_u: Vector3<f64> = sd.lu + fu.transpose() * v_x;
        let q_xx: Matrix5<f64> = sd.lxx + fx.transpose() * v_xx * fx;
        let q_uu_raw: Matrix3<f64> = sd.luu + fu.transpose() * v_xx * fu;
        let q_uu: Matrix3<f64> = q_uu_raw + reg * Matrix3::<f64>::identity();
        let q_ux: Matrix3x5<f64> = sd.lux + fu.transpose() * v_xx * fx;

        let q_uu_sym = 0.5 * (q_uu + q_uu.transpose());
        let q_uu_inv = q_uu_sym.try_inverse()?;

        if q_uu_inv.trace() <= 0.0 || q_uu_sym.determinant() <= 0.0 {
            return None;
        }

        let k_vec: Vector3<f64> = -q_uu_inv * q_u;
        let kk: Matrix3x5<f64> = -q_uu_inv * q_ux;

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
        let dx: Vector5<f64> = new_states[k] - prev_states[k];
        let u_new = prev_controls[k] + alpha * back.k_fb[k] + back.kk_fb[k] * dx;
        new_controls.push(u_new);
        cost += stage_cost_scalar(&new_states[k], &u_new, &u_prev, target, k as f64 * dt);
        let x_next = step(&new_states[k], &u_new, dt, params);
        new_states.push(x_next);
        u_prev = u_new;
    }
    (new_states, new_controls, cost)
}

fn run_ilqr(
    state: &RobotState,
    u_init: &[Control],
    u_prev_first: &Control,
    target: &MpcTarget,
    params: &RobotParams,
    cfg: &SolverConfig,
) -> SolveResult {
    let x0 = state.to_state();
    let mut controls = u_init.to_vec();
    let (mut states, mut cost) = rollout(&x0, &controls, u_prev_first, target, params, cfg.dt);
    let mut reg = cfg.reg_init.max(cfg.reg_min);
    let mut converged = false;
    let mut iters = 0;
    let alphas = [1.0, 0.5, 0.25, 0.125, 0.0625];

    let started = std::time::Instant::now();
    for it in 0..cfg.max_iters {
        iters = it + 1;
        let back = loop {
            match backward_pass(&states, &controls, u_prev_first, target, params, cfg.dt, reg) {
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
                alpha, &x0, &states, &controls, u_prev_first, &back, target, params, cfg.dt,
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
/// Returns whichever converges to lower cost. The robot's current heading is
/// carried in `state` and is the seed for both initialisations (no rotation
/// commanded until the solver finds a reason to turn).
pub fn solve(
    state: RobotState,
    target: &MpcTarget,
    params: &RobotParams,
    warm_start: Option<&Trajectory>,
    cfg: &SolverConfig,
) -> SolveResult {
    let n = cfg.horizon;
    // Hold-heading control: zero translational command, heading setpoint = the
    // current heading (so the heading lag and `(θ_cmd − θ)²` terms are inert).
    let hold = Control::new(0.0, 0.0, state.heading);
    let u_prev_first = warm_start
        .and_then(|t| t.controls.first().copied())
        .unwrap_or(hold);

    // Init 1: warm-start (shifted by one stage), padded with the last commanded
    // value. Falls back to hold-heading if no warm start available.
    let warm_controls: Vec<Control> = if let Some(ws) = warm_start {
        let mut v = Vec::with_capacity(n);
        for k in 0..n {
            let idx = (k + 1).min(ws.controls.len().saturating_sub(1));
            let c = ws.controls.get(idx).copied().unwrap_or(hold);
            v.push(c);
        }
        v
    } else {
        vec![hold; n]
    };

    // Init 2: constant velocity that would reach the target in `horizon · dt`,
    // holding the current heading. Magnitude clipped so we don't start from a
    // wildly large iterate.
    let horizon_time = cfg.dt * n as f64;
    let dir = target.p - state.pos;
    let desired = if horizon_time > 1.0e-9 {
        dir / horizon_time
    } else {
        Vec2::zeros()
    };
    let clipped = clip_speed(desired);
    let straight_controls: Vec<Control> = vec![Control::new(clipped.x, clipped.y, state.heading); n];

    // Candidate initialisations: warm-start, straight line, and (only when a
    // robot blocks the straight path) a left/right "go around" seed each. iLQR
    // runs from each; the lowest-cost result wins. With no blocking obstacle
    // this is exactly the previous two-init behaviour.
    let mut candidates: Vec<Vec<Control>> = Vec::with_capacity(4);
    candidates.push(warm_controls);
    candidates.push(straight_controls);
    candidates.extend(detour_seeds(
        state.pos,
        target.p,
        state.heading,
        &target.obstacles,
        n,
        cfg.dt,
    ));

    let mut best: Option<SolveResult> = None;
    for seed in &candidates {
        let r = run_ilqr(&state, seed, &u_prev_first, target, params, cfg);
        best = match best {
            Some(b) if b.final_cost <= r.final_cost => Some(b),
            _ => Some(r),
        };
    }
    best.expect("warm + straight candidates always present")
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
            heading: 0.0,
        };
        let r = solve(state, &target, &params(), None, &cfg);
        let u0 = r.trajectory.controls[0];
        // Translational command at rest at target must be ~0; heading setpoint
        // should hold the current heading (θ_cmd ≈ θ).
        let u_trans = (u0[0] * u0[0] + u0[1] * u0[1]).sqrt();
        assert!(
            u_trans < 1.0,
            "control at target should be ~0 mm/s, got {}",
            u_trans
        );
        assert!(
            (u0[2] - state.heading).abs() < 1.0e-2,
            "heading setpoint should hold, got {}",
            u0[2]
        );
    }

    #[test]
    fn converges_to_static_target() {
        let cfg = cfg_long();
        let target = MpcTarget::goto(Vec2::new(2000.0, 0.0));
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        };
        let r = solve(state, &target, &params(), None, &cfg);
        // Final state in the planned trajectory should be near the target with
        // small terminal velocity.
        let last = r.trajectory.states.last().unwrap();
        let p_err = ((last[0] - target.p.x).powi(2) + (last[1] - target.p.y).powi(2)).sqrt();
        let v_err = (last[2] * last[2] + last[3] * last[3]).sqrt();
        assert!(p_err < 200.0, "terminal position error {} mm", p_err);
        assert!(v_err < 200.0, "terminal velocity {} mm/s", v_err);
    }

    #[test]
    fn converges_to_target_heading() {
        // With a heading weight and a desired heading, the planned trajectory's
        // terminal heading should reach the target heading.
        let cfg = cfg_long();
        let mut target = MpcTarget::goto(Vec2::new(1000.0, 0.0));
        target.heading = 1.2;
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        };
        let r = solve(state, &target, &params(), None, &cfg);
        let last = r.trajectory.states.last().unwrap();
        assert!(
            (last[4] - target.heading).abs() < 0.1,
            "terminal heading {} should reach {}",
            last[4],
            target.heading
        );
    }

    #[test]
    fn bends_trajectory_around_head_on_obstacle() {
        use crate::obstacle::{Obstacle, ObstacleShape};
        use crate::types::ObstacleConfig;
        // The hard case: start, obstacle, and target are *exactly collinear*. A
        // purely radial barrier has no lateral gradient here, so without the
        // homotopy detour seeds the planner sits on the saddle and drives
        // straight through. With them, it must pick a side and clear the
        // keep-out. Uses the *default* obstacle config and the realistic in-app
        // solver config (short horizon, vision dt) so this guards the shipped
        // configuration, not a favourable one.
        let oc_cfg = ObstacleConfig::default();
        let cfg = SolverConfig {
            horizon: 50,
            dt: 0.016,
            ..SolverConfig::default()
        };
        let keepout = 2.0 * 90.0 + oc_cfg.robot_clearance;
        let oc = Vec2::new(1000.0, 0.0);
        let mut target = MpcTarget::goto(Vec2::new(3000.0, 0.0));
        target.obstacles = vec![Obstacle::fixed(
            ObstacleShape::Circle {
                center: oc,
                radius: keepout,
            },
            oc_cfg.weight,
            oc_cfg.influence,
        )];
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::new(1500.0, 0.0),
            heading: 0.0,
        };
        let r = solve(state, &target, &params(), None, &cfg);

        let min_clear = r
            .trajectory
            .states
            .iter()
            .map(|s| ((s[0] - oc.x).powi(2) + (s[1] - oc.y).powi(2)).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_clear > keepout,
            "head-on trajectory must clear the keep-out ({keepout} mm), closest approach was {min_clear} mm"
        );
        // And it must drive past the obstacle, not stall in front of it.
        let last = r.trajectory.states.last().unwrap();
        assert!(last[0] > 1400.0, "should drive past the obstacle, got x={}", last[0]);
    }

    #[test]
    #[ignore]
    fn diag_obstacle_sweep() {
        use crate::obstacle::{Obstacle, ObstacleShape};
        let keepout = 2.0 * 90.0 + 80.0; // 260 mm (live robot_clearance = 80)
        let target_p = Vec2::new(3000.0, 0.0);
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::new(2000.0, 0.0), // already moving fast toward target
            heading: 0.0,
        };
        // Mirror the user's live dies-settings.json so the sweep reflects reality.
        let mut live = RobotParams::default_hand_tuned();
        live.tau = [0.0, 0.0];
        live.accel_max = [10000.0, 10000.0];
        live.weights.position = 1.0;
        live.weights.velocity = 0.0;
        live.weights.control = 0.0;
        live.weights.control_smoothness = 0.0;
        // Real in-app config is horizon 50 at vision dt (~0.8 s lookahead).
        for &(horizon, dt) in &[(50usize, 0.016)] {
          eprintln!("--- horizon={horizon} dt={dt} (lookahead {:.2}s), pos_w={} ---", horizon as f64 * dt, live.weights.position);
          let cfg = SolverConfig { horizon, dt, ..SolverConfig::default() };
          for &(w, infl) in &[(2.0, 250.0), (2.0, 500.0), (10.0, 400.0), (40.0, 400.0), (40.0, 600.0)] {
            for &b in &[0.0, 150.0, 400.0] {
                let oc = Vec2::new(1000.0, b);
                let mut t = MpcTarget::goto(target_p);
                t.weights = live.weights.clone();
                t.obstacles = vec![Obstacle::fixed(
                    ObstacleShape::Circle { center: oc, radius: keepout },
                    w,
                    infl,
                )];
                // two solves: cold, then warm (mimic steady state)
                let cold = solve(state, &t, &live, None, &cfg);
                let warm = solve(state, &t, &live, Some(&cold.trajectory), &cfg);
                let min_clear = warm
                    .trajectory
                    .states
                    .iter()
                    .map(|s| ((s[0] - oc.x).powi(2) + (s[1] - oc.y).powi(2)).sqrt())
                    .fold(f64::INFINITY, f64::min);
                let last = warm.trajectory.states.last().unwrap();
                let cross = if min_clear < keepout { "CROSS" } else { "ok" };
                eprintln!(
                    "w={:>5} infl={:>5.0} b={:>5.0} -> min_clear={:>7.1} ({}) reach_x={:>7.0}",
                    w, infl, b, min_clear, cross, last[0]
                );
            }
          }
        }
    }

    #[test]
    fn warm_start_does_not_regress() {
        let cfg = SolverConfig::default();
        let target = MpcTarget::goto(Vec2::new(1500.0, 0.0));
        let state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        };
        let cold = solve(state, &target, &params(), None, &cfg);
        let warm = solve(state, &target, &params(), Some(&cold.trajectory), &cfg);
        // Warm start should at least not be worse than cold by a meaningful margin.
        assert!(
            warm.final_cost <= cold.final_cost + 1.0,
            "warm cost {} should not exceed cold cost {} by much",
            warm.final_cost,
            cold.final_cost
        );
    }
}
