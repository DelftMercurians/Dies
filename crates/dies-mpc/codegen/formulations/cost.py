"""Stage cost: quadratic translational tracking + smooth heading tracking.

L = ½·w_pos·‖pos−tp‖² + ½·w_vel·‖vel−tv‖²
  + ½·w_ctrl·‖u_trans‖² + ½·w_dctrl·‖u_trans − u_prev_trans‖²
  + w_yaw·(1 − cos(theta − theta_d))
  + ½·w_yawctrl·(theta_cmd − theta)²

The translational control terms apply to `u_trans = [vx_cmd, vy_cmd]` only — the
heading setpoint `theta_cmd` is regularised by the `(theta_cmd − theta)²` turn
term instead (absolute heading has no meaningful magnitude). Heading tracking
uses the wrap-aware `1 − cos` residual so the target heading attracts the short
way around.
"""

import sympy as sp

from ilqr_codegen import Model, half

m = Model("cost", title=__doc__)

# ── Rust function args ──────────────────────────────────────────────
m.arg("x", "&Vector5<f64>")
m.arg("u", "&Vector3<f64>")
m.arg("u_prev", "&Vector3<f64>")
m.arg("target", "&MpcTarget")

# ── Symbol bindings ─────────────────────────────────────────────────
x = m.vec("x", "px py vx vy theta", source="x[{i}]")
u = m.vec("u", "ux uy theta_cmd", source="u[{i}]")
u_prev = m.vec("u_prev", "upx upy uptheta", source="u_prev[{i}]")
target_p = m.vec("target_p", "target_px target_py", source=["target.p.x", "target.p.y"])
target_v = m.vec("target_v", "target_vx target_vy", source=["target.v.x", "target.v.y"])
target_theta = m.scalar("target_theta", "target.heading")

w = m.scalars(
    {
        "w_pos": "target.weights.position",
        "w_vel": "target.weights.velocity",
        "w_ctrl": "target.weights.control",
        "w_dctrl": "target.weights.control_smoothness",
        "w_yaw": "target.weights.heading",
        "w_yawctrl": "target.weights.heading_control",
    }
)

# ── Model ───────────────────────────────────────────────────────────
pos, vel = x[:2, :], x[2:4, :]
theta = x[4]
u_trans = u[:2, :]
uprev_trans = u_prev[:2, :]
theta_cmd = u[2]
du = u_trans - uprev_trans

L = m.eq(
    "L",
    (
        half * w.w_pos * (pos - target_p).dot(pos - target_p)
        + half * w.w_vel * (vel - target_v).dot(vel - target_v)
        + half * w.w_ctrl * u_trans.dot(u_trans)
        + half * w.w_dctrl * du.dot(du)
        + w.w_yaw * (1 - sp.cos(theta - target_theta))
        + half * w.w_yawctrl * (theta_cmd - theta) ** 2
    ),
)

# ── Stage exports ───────────────────────────────────────────────────
m.export("cost", L, rust_type="f64")
m.export("lx", m.jac(L, x), rust_type="Vector5<f64>")
m.export("lu", m.jac(L, u), rust_type="Vector3<f64>")
m.export("lxx", m.hessian(L, x), rust_type="Matrix5<f64>")
m.export("luu", m.hessian(L, u), rust_type="Matrix3<f64>")
m.export("lux", m.jac(m.jac(L, u), x), rust_type="Matrix3x5<f64>")

# ── Functions ───────────────────────────────────────────────────────
m.function("stage_derivs", returns=["cost", "lx", "lu", "lxx", "luu", "lux"])
m.function("stage_cost_scalar", returns=["cost"])
