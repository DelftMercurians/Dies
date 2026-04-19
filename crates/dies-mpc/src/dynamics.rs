//! Robot translational dynamics.
//!
//! State `x = [px, py, vx, vy]` in the global frame.
//! Control `u = [vx_cmd, vy_cmd]` in the global frame.
//! Heading `θ` is an exogenous scalar input per step (supplied by the caller —
//! the onboard IMU/magnetometer controller handles heading outside MPC).
//!
//! Continuous body-axis dynamics per axis `i ∈ {FWD, STRAFE}`:
//! ```text
//! v̇_b[i] =  a_max[i]   · tanh( (v_cmd_b[i] − v_b[i]) / (τ[i] · a_max[i]) )
//!         − stiction[i] · tanh(  v_b[i] / v_ε )
//! ```
//! The first term is a smooth-saturated velocity-lag response. The second is
//! a smoothed stiction opposing motion at low speeds — this is what the MTP
//! controller lacks and why it struggles with small-amplitude moves.
//!
//! We advance with forward Euler (`dt = 60 ms`, `τ ≈ 80–100 ms`, marginally
//! stable but fine for an MPC predictor; iLQR needs correct gradients more
//! than high-fidelity numerics). Jacobians are derived analytically.

use crate::types::{Control, ControlJac, Mat2, RobotParams, State, StateJac, Vec2, FWD, STRAFE};

/// Continuous-time state derivative and its Jacobians, evaluated at `(x, u, θ)`.
///
/// Returns `(ẋ, ∂ẋ/∂x, ∂ẋ/∂u)`.
fn continuous(
    x: &State,
    u: &Control,
    heading: f64,
    p: &RobotParams,
) -> (State, StateJac, ControlJac) {
    let v_global = Vec2::new(x[2], x[3]);
    let (c, s) = (heading.cos(), heading.sin());
    let r = Mat2::new(c, -s, s, c);
    let rt = r.transpose();

    let v_body = rt * v_global;
    let cmd_body = rt * u;

    let mut a_body = Vec2::zeros();
    let mut da_dv_body = Mat2::zeros();
    let mut da_du_body = Mat2::zeros();

    for axis in [FWD, STRAFE] {
        let tau = p.tau[axis];
        let a_max = p.a_max[axis];
        let stiction = p.stiction[axis];
        let v = v_body[axis];
        let vcmd = cmd_body[axis];

        // Accel term: a_max · tanh(err / (τ·a_max))
        let err = vcmd - v;
        let arg_acc = err / (tau * a_max);
        let t_acc = arg_acc.tanh();
        let sech2_acc = 1.0 - t_acc * t_acc;
        let accel = a_max * t_acc;
        // ∂accel/∂err = sech²(arg) / τ
        let d_accel_d_err = sech2_acc / tau;

        // Stiction term: −stiction · tanh(v / v_ε)
        let arg_stic = v / p.v_eps;
        let t_stic = arg_stic.tanh();
        let sech2_stic = 1.0 - t_stic * t_stic;
        let stic = -stiction * t_stic;
        // ∂stic/∂v = −stiction · sech²(v/v_ε) / v_ε
        let d_stic_d_v = -stiction * sech2_stic / p.v_eps;

        a_body[axis] = accel + stic;
        // err = vcmd - v ⇒ ∂err/∂v = −1, ∂err/∂vcmd = +1
        da_dv_body[(axis, axis)] = -d_accel_d_err + d_stic_d_v;
        da_du_body[(axis, axis)] = d_accel_d_err;
    }

    // Rotate body-frame accelerations and their Jacobians back to the global frame.
    let a_global = r * a_body;
    let da_dv_global = r * da_dv_body * rt;
    let da_du_global = r * da_du_body * rt;

    let f = State::new(v_global.x, v_global.y, a_global.x, a_global.y);

    let mut fx = StateJac::zeros();
    fx[(0, 2)] = 1.0;
    fx[(1, 3)] = 1.0;
    for i in 0..2 {
        for j in 0..2 {
            fx[(2 + i, 2 + j)] = da_dv_global[(i, j)];
        }
    }

    let mut fu = ControlJac::zeros();
    for i in 0..2 {
        for j in 0..2 {
            fu[(2 + i, j)] = da_du_global[(i, j)];
        }
    }

    (f, fx, fu)
}

/// One forward-Euler dynamics step.
pub fn step(x: &State, u: &Control, heading: f64, dt: f64, p: &RobotParams) -> State {
    let (f, _, _) = continuous(x, u, heading, p);
    x + f * dt
}

/// One Euler step together with analytic discrete-time Jacobians
/// `(F_x = ∂x_next/∂x, F_u = ∂x_next/∂u)`.
pub fn step_with_jacobians(
    x: &State,
    u: &Control,
    heading: f64,
    dt: f64,
    p: &RobotParams,
) -> (State, StateJac, ControlJac) {
    let (f, fx_c, fu_c) = continuous(x, u, heading, p);
    let fx = StateJac::identity() + fx_c * dt;
    let fu = fu_c * dt;
    (x + f * dt, fx, fu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RobotState;
    use approx::assert_abs_diff_eq;

    fn params() -> RobotParams {
        RobotParams::default_hand_tuned()
    }

    #[test]
    fn zero_cmd_at_rest_stays_put() {
        let p = params();
        let mut x = State::zeros();
        let u = Control::zeros();
        let dt = 0.06;
        for _ in 0..20 {
            x = step(&x, &u, 0.0, dt, &p);
        }
        assert!(x.norm() < 1.0e-6);
    }

    #[test]
    fn stiction_suppresses_small_cmd_vs_no_stiction() {
        // With stiction present, a small commanded velocity settles at an
        // equilibrium well below the commanded value. Without stiction, the
        // robot reaches (essentially) the commanded velocity after a few τ.
        let p_with = params();
        let p_without = RobotParams {
            stiction: [0.0, 0.0],
            ..params()
        };
        let cmd = 10.0;
        let u = Control::new(cmd, 0.0);
        let dt = 0.01;
        let mut x_with = State::zeros();
        let mut x_without = State::zeros();
        for _ in 0..200 {
            x_with = step(&x_with, &u, 0.0, dt, &p_with);
            x_without = step(&x_without, &u, 0.0, dt, &p_without);
        }
        let v_with = x_with[2];
        let v_without = x_without[2];
        // Without stiction: close to commanded. With stiction: suppressed.
        assert!(
            v_without > 0.9 * cmd,
            "no-stiction velocity should track cmd, got {}",
            v_without
        );
        assert!(
            v_with < 0.8 * v_without,
            "stiction should hold velocity below 80% of no-stiction case; got {} vs {}",
            v_with,
            v_without
        );
    }

    #[test]
    fn large_cmd_converges_to_cmd_velocity() {
        let p = params();
        let mut x = State::zeros();
        let target_v = 2000.0;
        let u = Control::new(target_v, 0.0);
        let dt = 0.01;
        // 3τ ≈ 0.24 s. Simulate longer and assert we've reached near target.
        for _ in 0..100 {
            x = step(&x, &u, 0.0, dt, &p);
        }
        let st = RobotState::from_state(&x);
        let tol = 0.05 * target_v; // 5%
        assert!(
            (st.vel.x - target_v).abs() < tol,
            "vx = {} did not converge to {} within {}",
            st.vel.x,
            target_v,
            tol
        );
        assert!(st.vel.y.abs() < 10.0);
    }

    #[test]
    fn jacobians_match_finite_diff() {
        let p = params();
        let dt = 0.06;
        let samples: &[(State, Control, f64)] = &[
            (
                State::new(0.0, 0.0, 0.0, 0.0),
                Control::new(500.0, 0.0),
                0.0,
            ),
            (
                State::new(100.0, -50.0, 300.0, -200.0),
                Control::new(1500.0, 800.0),
                0.7,
            ),
            (
                State::new(-500.0, 400.0, -1200.0, 1500.0),
                Control::new(-2000.0, 1800.0),
                -1.3,
            ),
            (
                State::new(0.0, 0.0, 5.0, -3.0),
                Control::new(10.0, 8.0),
                2.4,
            ),
        ];
        for (x, u, h) in samples {
            let (_, fx, fu) = step_with_jacobians(x, u, *h, dt, &p);
            let eps = 1.0e-4;
            // ∂x_next/∂x
            for j in 0..4 {
                let mut xp = *x;
                let mut xm = *x;
                xp[j] += eps;
                xm[j] -= eps;
                let sp = step(&xp, u, *h, dt, &p);
                let sm = step(&xm, u, *h, dt, &p);
                let fd = (sp - sm) / (2.0 * eps);
                for i in 0..4 {
                    assert_abs_diff_eq!(fx[(i, j)], fd[i], epsilon = 1.0e-5);
                }
            }
            // ∂x_next/∂u
            for j in 0..2 {
                let mut up = *u;
                let mut um = *u;
                up[j] += eps;
                um[j] -= eps;
                let sp = step(x, &up, *h, dt, &p);
                let sm = step(x, &um, *h, dt, &p);
                let fd = (sp - sm) / (2.0 * eps);
                for i in 0..4 {
                    assert_abs_diff_eq!(fu[(i, j)], fd[i], epsilon = 1.0e-5);
                }
            }
        }
    }

    #[test]
    fn body_frame_rotation_is_consistent() {
        // With the forward axis aligned to world-x (heading = 0), commanding
        // world-x velocity produces the same speed-up as commanding world-y
        // when heading = π/2 (forward axis now points along world-y).
        let p = params();
        let dt = 0.01;
        let mut x0 = State::zeros();
        let u0 = Control::new(1500.0, 0.0);
        for _ in 0..30 {
            x0 = step(&x0, &u0, 0.0, dt, &p);
        }

        let mut x1 = State::zeros();
        let u1 = Control::new(0.0, 1500.0);
        for _ in 0..30 {
            x1 = step(&x1, &u1, std::f64::consts::FRAC_PI_2, dt, &p);
        }

        assert_abs_diff_eq!(x0[2], x1[3], epsilon = 1.0e-6);
        assert_abs_diff_eq!(x0[3].abs(), x1[2].abs(), epsilon = 1.0e-6);
    }
}
