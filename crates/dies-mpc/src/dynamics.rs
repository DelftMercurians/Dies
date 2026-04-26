//! Robot translational dynamics — first-order velocity lag per body axis.
//!
//! State `x = [px, py, vx, vy]` global frame.
//! Control `u = [vx_cmd, vy_cmd]` global frame.
//! Heading `θ` is exogenous per stage (handled by the onboard yaw controller,
//! supplied by the caller as a per-stage trajectory).
//!
//! Body-axis dynamics, per axis `i ∈ {FWD, STRAFE}`:
//! ```text
//! v̇_b[i] = (v_cmd_b[i] − v_b[i]) / τ[i]
//! ```
//! Two parameters total. Linear in body frame; rotation by heading `θ` makes
//! the global-frame dynamics nonlinear in `(x, u)` only through the heading
//! input. We integrate with forward Euler at the MPC stage `dt`.

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
        let inv_tau = 1.0 / p.tau[axis];
        let a_max = p.accel_max[axis];
        let raw = (cmd_body[axis] - v_body[axis]) * inv_tau;
        // Smooth saturation: a = a_max · tanh(raw / a_max). Linear near zero
        // (a ≈ raw), asymptotes to ±a_max far from steady state.
        let t = (raw / a_max).tanh();
        // sech²(x) = 1 − tanh²(x); chain rule through raw.
        let sech2 = 1.0 - t * t;
        a_body[axis] = a_max * t;
        da_dv_body[(axis, axis)] = -inv_tau * sech2;
        da_du_body[(axis, axis)] = inv_tau * sech2;
    }

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
        assert!(x.norm() < 1.0e-9);
    }

    #[test]
    fn step_response_reaches_command() {
        // First-order lag: v(t) = v_cmd · (1 − exp(−t/τ)). After 5τ we should
        // be within 2% of the commanded velocity. Target stays in the linear
        // regime — `target_v / τ < a_max` — so the tanh saturation is inert.
        let p = params();
        let dt = 0.001;
        let target_v = 200.0;
        assert!(target_v / p.tau[FWD] < p.accel_max[FWD]);
        let u = Control::new(target_v, 0.0);
        let mut x = State::zeros();
        let n = (5.0 * p.tau[FWD] / dt) as usize;
        for _ in 0..n {
            x = step(&x, &u, 0.0, dt, &p);
        }
        let st = RobotState::from_state(&x);
        assert!(
            (st.vel.x - target_v).abs() < 0.02 * target_v,
            "vx = {} did not reach {} within 2%",
            st.vel.x,
            target_v
        );
        assert!(st.vel.y.abs() < 1.0);
    }

    #[test]
    fn acceleration_is_capped_at_a_max() {
        // Far from steady state the realised accel must be bounded by a_max
        // (small ε for the tanh approach to the asymptote).
        let p = params();
        let dt = 0.001;
        let u = Control::new(10_000.0, 0.0);
        let x = State::zeros();
        let x_next = step(&x, &u, 0.0, dt, &p);
        let dv = (x_next[2] - x[2]).abs();
        let a_observed = dv / dt;
        assert!(
            a_observed <= p.accel_max[FWD] + 1.0e-6,
            "observed accel {} exceeded a_max {}",
            a_observed,
            p.accel_max[FWD]
        );
        // And it should be close to the ceiling, not far below.
        assert!(a_observed > 0.99 * p.accel_max[FWD]);
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
        ];
        for (x, u, h) in samples {
            let (_, fx, fu) = step_with_jacobians(x, u, *h, dt, &p);
            let eps = 1.0e-4;
            for j in 0..4 {
                let mut xp = *x;
                let mut xm = *x;
                xp[j] += eps;
                xm[j] -= eps;
                let sp = step(&xp, u, *h, dt, &p);
                let sm = step(&xm, u, *h, dt, &p);
                let fd = (sp - sm) / (2.0 * eps);
                for i in 0..4 {
                    assert_abs_diff_eq!(fx[(i, j)], fd[i], epsilon = 1.0e-7);
                }
            }
            for j in 0..2 {
                let mut up = *u;
                let mut um = *u;
                up[j] += eps;
                um[j] -= eps;
                let sp = step(x, &up, *h, dt, &p);
                let sm = step(x, &um, *h, dt, &p);
                let fd = (sp - sm) / (2.0 * eps);
                for i in 0..4 {
                    assert_abs_diff_eq!(fu[(i, j)], fd[i], epsilon = 1.0e-7);
                }
            }
        }
    }

    #[test]
    fn body_frame_rotation_is_consistent() {
        // Forward-axis cmd at heading=0 should produce the same speed as the
        // same magnitude commanded along strafe at heading=π/2.
        let p = params();
        let dt = 0.001;
        let mut x0 = State::zeros();
        let u0 = Control::new(1500.0, 0.0);
        for _ in 0..300 {
            x0 = step(&x0, &u0, 0.0, dt, &p);
        }

        let mut x1 = State::zeros();
        let u1 = Control::new(0.0, 1500.0);
        for _ in 0..300 {
            x1 = step(&x1, &u1, std::f64::consts::FRAC_PI_2, dt, &p);
        }

        assert_abs_diff_eq!(x0[2], x1[3], epsilon = 1.0e-6);
        assert_abs_diff_eq!(x0[3].abs(), x1[2].abs(), epsilon = 1.0e-6);
    }
}
