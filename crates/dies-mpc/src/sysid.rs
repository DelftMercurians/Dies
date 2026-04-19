//! Offline system-identification fit for the 7 dynamics parameters.
//!
//! Feed this function a sequence of `(t, cmd, heading, observed_state)` samples
//! from a field calibration routine; it runs Levenberg-Marquardt over the
//! log-transformed parameters (all 7 are strictly positive) and returns a
//! fitted `RobotParams` plus per-axis velocity residual RMS.
//!
//! The residual is a *one-step velocity prediction error*: for each
//! consecutive pair, forward-integrate from the earlier sample's state with
//! its commanded velocity for the intervening `Δt` and compare the
//! predicted velocity against the next sample's observed velocity. We
//! deliberately ignore position residuals — they are integrated velocity
//! residuals and add no independent information while biasing the fit
//! toward slow-varying errors.

use nalgebra::{DMatrix, DVector, Matrix2, SMatrix, Vector2};

use crate::types::{FitOptions, FitResult, RobotParams, Sample, FWD, STRAFE};

const N_PARAMS: usize = 7;

fn to_log(p: &RobotParams) -> [f64; N_PARAMS] {
    let a = p.to_array();
    let mut out = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        out[i] = a[i].max(1.0e-12).ln();
    }
    out
}

fn from_log(theta: &[f64; N_PARAMS]) -> RobotParams {
    let mut a = [0.0; N_PARAMS];
    for i in 0..N_PARAMS {
        a[i] = theta[i].exp();
    }
    RobotParams::from_array(a)
}

/// At sample `i`, compute both the predicted next velocity and the Jacobian
/// of that prediction with respect to each of the 7 raw parameters.
fn predicted_vel_and_param_jacobian(
    sample: &Sample,
    dt: f64,
    p: &RobotParams,
) -> (Vector2<f64>, SMatrix<f64, 2, 7>) {
    let v_global = Vector2::new(sample.state.vel.x, sample.state.vel.y);
    let (c, s) = (sample.heading.cos(), sample.heading.sin());
    let r = Matrix2::new(c, -s, s, c);
    let rt = r.transpose();
    let v_body = rt * v_global;
    let cmd_body = rt * sample.cmd;

    let mut a_body = Vector2::zeros();
    let mut j_ab = SMatrix::<f64, 2, 7>::zeros();

    for axis in [FWD, STRAFE] {
        let tau = p.tau[axis];
        let a_max = p.a_max[axis];
        let stiction = p.stiction[axis];
        let v_eps = p.v_eps;
        let v = v_body[axis];
        let vcmd = cmd_body[axis];
        let err = vcmd - v;

        let arg_acc = err / (tau * a_max);
        let t_acc = arg_acc.tanh();
        let sech2_acc = 1.0 - t_acc * t_acc;
        let arg_stic = v / v_eps;
        let t_stic = arg_stic.tanh();
        let sech2_stic = 1.0 - t_stic * t_stic;

        a_body[axis] = a_max * t_acc - stiction * t_stic;

        // d a_body[axis] / d params:
        let d_dtau = -err * sech2_acc / (tau * tau);
        let d_damax = t_acc - err * sech2_acc / (tau * a_max);
        let d_dstic = -t_stic;
        let d_dveps = stiction * v * sech2_stic / (v_eps * v_eps);

        let idx_tau = axis; //   0 or 1
        let idx_amax = 2 + axis; // 2 or 3
        let idx_stic = 4 + axis; // 4 or 5
        j_ab[(axis, idx_tau)] = d_dtau;
        j_ab[(axis, idx_amax)] = d_damax;
        j_ab[(axis, idx_stic)] = d_dstic;
        j_ab[(axis, 6)] = d_dveps;
    }

    let a_global = r * a_body;
    let j_ag = r * j_ab; // 2x7 — ∂a_global/∂param
    let v_pred = v_global + a_global * dt;
    let j_vpred = j_ag * dt;
    (v_pred, j_vpred)
}

/// Build the stacked residual vector and parameter Jacobian over all sample
/// pairs. Residual shape: `2·(N−1)`. Jacobian shape: `2·(N−1) × 7`.
fn build_residual_and_jacobian(
    samples: &[Sample],
    p: &RobotParams,
) -> (DVector<f64>, DMatrix<f64>) {
    let pairs = samples.len().saturating_sub(1);
    let mut r = DVector::<f64>::zeros(2 * pairs);
    let mut j = DMatrix::<f64>::zeros(2 * pairs, N_PARAMS);
    for i in 0..pairs {
        let dt = (samples[i + 1].t - samples[i].t).max(1.0e-6);
        let (v_pred, j_block) = predicted_vel_and_param_jacobian(&samples[i], dt, p);
        let v_obs = Vector2::new(samples[i + 1].state.vel.x, samples[i + 1].state.vel.y);
        let res = v_obs - v_pred; // residual = observed − predicted
        r[2 * i] = res.x;
        r[2 * i + 1] = res.y;
        // ∂r/∂param = −∂v_pred/∂param
        for k in 0..N_PARAMS {
            j[(2 * i, k)] = -j_block[(0, k)];
            j[(2 * i + 1, k)] = -j_block[(1, k)];
        }
    }
    (r, j)
}

/// Apply `dp/dθ = p` to convert a raw-parameter Jacobian into a log-space
/// Jacobian. This is just column scaling.
fn log_jacobian(j: &DMatrix<f64>, p: &RobotParams) -> DMatrix<f64> {
    let arr = p.to_array();
    let mut jl = j.clone();
    for k in 0..N_PARAMS {
        let scale = arr[k];
        for i in 0..jl.nrows() {
            jl[(i, k)] *= scale;
        }
    }
    jl
}

fn cost(r: &DVector<f64>) -> f64 {
    r.norm_squared()
}

/// Levenberg-Marquardt fit over log-transformed parameters.
pub fn fit_params(samples: &[Sample], init: RobotParams, opts: FitOptions) -> FitResult {
    if samples.len() < 2 {
        return FitResult {
            params: init,
            residual_rms_per_axis: [0.0, 0.0],
            iters: 0,
            converged: true,
        };
    }

    let mut theta = to_log(&init);
    let mut p = from_log(&theta);
    let (mut r, mut j_raw) = build_residual_and_jacobian(samples, &p);
    let mut c = cost(&r);
    let mut lambda = opts.lambda_init;
    let mut converged = false;
    let mut iters = 0;

    for it in 0..opts.max_iters {
        iters = it + 1;
        let j_log = log_jacobian(&j_raw, &p);
        let jt_j: DMatrix<f64> = j_log.transpose() * &j_log;
        let jt_r: DVector<f64> = j_log.transpose() * &r;

        // Marquardt damping with diagonal scaling
        let mut a = jt_j.clone();
        for i in 0..N_PARAMS {
            let diag_i = jt_j[(i, i)];
            a[(i, i)] += lambda * diag_i.max(1.0e-6);
        }
        let Some(a_inv) = a.try_inverse() else {
            lambda *= 10.0;
            if lambda > 1.0e20 {
                break;
            }
            continue;
        };
        let dtheta: DVector<f64> = -(a_inv * jt_r);

        let mut theta_new = theta;
        for i in 0..N_PARAMS {
            theta_new[i] += dtheta[i];
        }
        let p_new = from_log(&theta_new);
        let (r_new, j_new) = build_residual_and_jacobian(samples, &p_new);
        let c_new = cost(&r_new);

        if c_new < c {
            let rel = (c - c_new) / c.max(1.0e-12);
            theta = theta_new;
            p = p_new;
            r = r_new;
            j_raw = j_new;
            c = c_new;
            lambda = (lambda * 0.5).max(1.0e-12);
            if rel < opts.tol {
                converged = true;
                break;
            }
        } else {
            lambda *= 2.0;
            if lambda > 1.0e20 {
                break;
            }
        }
    }

    let n_pairs = samples.len() - 1;
    let mut rss_x = 0.0;
    let mut rss_y = 0.0;
    for i in 0..n_pairs {
        rss_x += r[2 * i].powi(2);
        rss_y += r[2 * i + 1].powi(2);
    }
    let rms_x = (rss_x / n_pairs as f64).sqrt();
    let rms_y = (rss_y / n_pairs as f64).sqrt();

    FitResult {
        params: p,
        residual_rms_per_axis: [rms_x, rms_y],
        iters,
        converged,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dynamics::step;
    use crate::types::{RobotState, Vec2};
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn synthetic_samples(
        truth: &RobotParams,
        commands: &[(f64, Vec2, f64)], // per-step: (dt, cmd_to_apply, heading)
        noise_std: f64,
        seed: u64,
    ) -> Vec<Sample> {
        // Convention: `samples[i]` holds the state at time `t_i` and the
        // command + heading to be applied over the interval `[t_i, t_{i+1}]`.
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, noise_std).unwrap();
        let mut samples = Vec::with_capacity(commands.len() + 1);
        let mut t = 0.0;
        let mut state = RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
        };
        for (dt, cmd, heading) in commands {
            samples.push(Sample {
                t,
                cmd: *cmd,
                heading: *heading,
                state,
            });
            let xs = state.to_state();
            let x_next = step(&xs, cmd, *heading, *dt, truth);
            let mut next_state = RobotState::from_state(&x_next);
            next_state.vel.x += noise.sample(&mut rng);
            next_state.vel.y += noise.sample(&mut rng);
            state = next_state;
            t += dt;
        }
        // Trailing sample captures the final observed state; its cmd/heading
        // are inherited from the last interval (unused by the fit, since
        // there's no further observation to predict against).
        let last = commands.last().unwrap();
        samples.push(Sample {
            t,
            cmd: last.1,
            heading: last.2,
            state,
        });
        samples
    }

    fn rich_command_sequence() -> Vec<(f64, Vec2, f64)> {
        // Mixture of steps, ramps, and rotations through various headings
        // to excite all 7 parameters independently.
        let dt = 0.01;
        let mut cmds = Vec::new();
        // Step fwd
        for _ in 0..50 {
            cmds.push((dt, Vec2::new(2000.0, 0.0), 0.0));
        }
        // Step reverse
        for _ in 0..50 {
            cmds.push((dt, Vec2::new(-2000.0, 0.0), 0.0));
        }
        // Strafe (rotated: forward axis is world-y now)
        for _ in 0..50 {
            cmds.push((dt, Vec2::new(1500.0, 0.0), std::f64::consts::FRAC_PI_2));
        }
        // Slow ramp to exercise stiction regime
        for i in 0..80 {
            let mag = (i as f64) * 3.0;
            cmds.push((dt, Vec2::new(mag, 0.0), 0.0));
        }
        // Back to zero cmd at low speed
        for _ in 0..40 {
            cmds.push((dt, Vec2::new(0.0, 0.0), 0.0));
        }
        // Small strafe cmds near zero (stiction-dominated regime)
        for _ in 0..60 {
            cmds.push((dt, Vec2::new(0.0, 20.0), 0.0));
        }
        // Diagonal at intermediate speed
        for _ in 0..60 {
            cmds.push((dt, Vec2::new(1000.0, 1000.0), 0.3));
        }
        cmds
    }

    #[test]
    fn recovers_known_params_from_synthetic() {
        let truth = RobotParams {
            tau: [0.075, 0.11],
            a_max: [5500.0, 4200.0],
            stiction: [120.0, 180.0],
            v_eps: 25.0,
        };
        let cmds = rich_command_sequence();
        let samples = synthetic_samples(&truth, &cmds, 0.0, 1);
        // Deliberately bad starting point.
        let init = RobotParams {
            tau: [0.05, 0.15],
            a_max: [4000.0, 3000.0],
            stiction: [80.0, 250.0],
            v_eps: 15.0,
        };
        let fit = fit_params(&samples, init, FitOptions::default());
        assert!(fit.iters > 0);
        let p = &fit.params;
        for (axis, name) in [(FWD, "fwd"), (STRAFE, "strafe")] {
            assert_abs_diff_eq!(
                p.tau[axis],
                truth.tau[axis],
                epsilon = 0.05 * truth.tau[axis]
            );
            assert_abs_diff_eq!(
                p.a_max[axis],
                truth.a_max[axis],
                epsilon = 0.05 * truth.a_max[axis]
            );
            assert_abs_diff_eq!(
                p.stiction[axis],
                truth.stiction[axis],
                epsilon = 0.05 * truth.stiction[axis],
                // avoid unused variable warning in fail msg
            );
            let _ = name;
        }
        assert_abs_diff_eq!(p.v_eps, truth.v_eps, epsilon = 0.10 * truth.v_eps);
    }

    #[test]
    fn recovers_with_moderate_noise() {
        let truth = RobotParams {
            tau: [0.08, 0.10],
            a_max: [6000.0, 4500.0],
            stiction: [150.0, 200.0],
            v_eps: 30.0,
        };
        let cmds = rich_command_sequence();
        let samples = synthetic_samples(&truth, &cmds, 10.0, 42);
        let init = RobotParams::default_hand_tuned();
        let fit = fit_params(&samples, init, FitOptions::default());
        let p = &fit.params;
        for axis in [FWD, STRAFE] {
            assert_abs_diff_eq!(
                p.tau[axis],
                truth.tau[axis],
                epsilon = 0.15 * truth.tau[axis]
            );
            assert_abs_diff_eq!(
                p.a_max[axis],
                truth.a_max[axis],
                epsilon = 0.15 * truth.a_max[axis]
            );
            assert_abs_diff_eq!(
                p.stiction[axis],
                truth.stiction[axis],
                epsilon = 0.3 * truth.stiction[axis]
            );
        }
    }

    #[test]
    fn positive_params_stay_positive_with_bad_init() {
        let truth = RobotParams::default_hand_tuned();
        let cmds = rich_command_sequence();
        let samples = synthetic_samples(&truth, &cmds, 5.0, 99);
        // Intentionally terrible init — orders of magnitude off.
        let bad_init = RobotParams {
            tau: [0.001, 1.0],
            a_max: [100.0, 100_000.0],
            stiction: [1.0, 5000.0],
            v_eps: 0.5,
        };
        let fit = fit_params(&samples, bad_init, FitOptions::default());
        let p = &fit.params;
        assert!(p.tau[FWD] > 0.0 && p.tau[STRAFE] > 0.0);
        assert!(p.a_max[FWD] > 0.0 && p.a_max[STRAFE] > 0.0);
        assert!(p.stiction[FWD] >= 0.0 && p.stiction[STRAFE] >= 0.0);
        assert!(p.v_eps > 0.0);
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let truth = RobotParams::default_hand_tuned();
        let sample = Sample {
            t: 0.0,
            cmd: Vec2::new(1200.0, -300.0),
            heading: 0.4,
            state: RobotState {
                pos: Vec2::zeros(),
                vel: Vec2::new(500.0, -200.0),
            },
        };
        let dt = 0.02;
        let (_, j) = predicted_vel_and_param_jacobian(&sample, dt, &truth);
        let base = truth.to_array();
        let eps_rel = 1.0e-5;
        for k in 0..N_PARAMS {
            let mut bp = base;
            let mut bm = base;
            let e = eps_rel * base[k].abs().max(1.0);
            bp[k] += e;
            bm[k] -= e;
            let pp = RobotParams::from_array(bp);
            let pm = RobotParams::from_array(bm);
            let (v_p, _) = predicted_vel_and_param_jacobian(&sample, dt, &pp);
            let (v_m, _) = predicted_vel_and_param_jacobian(&sample, dt, &pm);
            let fd = (v_p - v_m) / (2.0 * e);
            assert_abs_diff_eq!(j[(0, k)], fd.x, epsilon = 1.0e-4 * fd.x.abs().max(1.0));
            assert_abs_diff_eq!(j[(1, k)], fd.y, epsilon = 1.0e-4 * fd.y.abs().max(1.0));
        }
    }
}
