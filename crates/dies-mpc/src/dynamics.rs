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

use crate::types::{Control, ControlJac, RobotParams, State, StateJac};

/// One forward-Euler dynamics step.
pub fn step(x: &State, u: &Control, heading: f64, dt: f64, p: &RobotParams) -> State {
    crate::generated::dynamics::step(x, u, heading, dt, p)
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
    crate::generated::dynamics::step_with_jacobians(x, u, heading, dt, p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RobotState, FWD};
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
