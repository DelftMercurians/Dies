"""Quadratic stage cost for the iLQR solver.

Stage L = ½·w_pos·‖pos−tp‖² + ½·w_vel·‖vel−tv‖² + ½·w_ctrl·‖u‖² + ½·w_dctrl·‖u−u_prev‖²
"""

from ilqr_codegen import Model, half

m = Model("cost", title=__doc__)

# ── Rust function args ──────────────────────────────────────────────
m.arg("x", "&Vector4<f64>")
m.arg("u", "&Vector2<f64>")
m.arg("u_prev", "&Vector2<f64>")
m.arg("target", "&MpcTarget")

# ── Symbol bindings ─────────────────────────────────────────────────
x = m.vec("x", "px py vx vy", source="x[{i}]")
u = m.vec("u", "ux uy", source="u[{i}]")
u_prev = m.vec("u_prev", "upx upy", source="u_prev[{i}]")
target_p = m.vec("target_p", "target_px target_py", source=["target.p.x", "target.p.y"])
target_v = m.vec("target_v", "target_vx target_vy", source=["target.v.x", "target.v.y"])

w = m.scalars(
    {
        "w_pos": "target.weights.position",
        "w_vel": "target.weights.velocity",
        "w_ctrl": "target.weights.control",
        "w_dctrl": "target.weights.control_smoothness",
    }
)

# ── Model ───────────────────────────────────────────────────────────
pos, vel = x[:2, :], x[2:, :]
du = u - u_prev

L = m.eq(
    "L",
    (
        half * w.w_pos * (pos - target_p).dot(pos - target_p)
        + half * w.w_vel * (vel - target_v).dot(vel - target_v)
        + half * w.w_ctrl * u.dot(u)
        + half * w.w_dctrl * du.dot(du)
    ),
)

# ── Stage exports ───────────────────────────────────────────────────
m.export("cost", L, rust_type="f64")
m.export("lx", m.jac(L, x), rust_type="Vector4<f64>")
m.export("lu", m.jac(L, u), rust_type="Vector2<f64>")
m.export("lxx", m.hessian(L, x), rust_type="Matrix4<f64>")
m.export("luu", m.hessian(L, u), rust_type="Matrix2<f64>")
m.export("lux", m.jac(m.jac(L, u), x), rust_type="Matrix2x4<f64>")

# ── Functions ───────────────────────────────────────────────────────
m.function("stage_derivs", returns=["cost", "lx", "lu", "lxx", "luu", "lux"])
m.function("stage_cost_scalar", returns=["cost"])
