"""Robot translational dynamics — first-order velocity lag per body axis.

State `x = [px, py, vx, vy]` global frame.
Control `u = [vx_cmd, vy_cmd]` global frame.
Heading θ is exogenous per stage. Forward-Euler integration at MPC stage dt.
"""

import sympy as sp

from ilqr_codegen import Model

m = Model("dynamics", title=__doc__)

# ── Rust function args ──────────────────────────────────────────────
m.arg("x", "&State")
m.arg("u", "&Control")
m.arg("heading", "f64")
m.arg("dt", "f64")
m.arg("p", "&RobotParams")

# ── Symbol bindings (sym → Rust scalar source) ──────────────────────
x = m.vec("x", "px py vx vy", source="x[{i}]")
u = m.vec("u", "ux uy", source="u[{i}]")
heading = m.scalar("heading")
dt = m.scalar("dt")
tau = m.vec("tau", "tau_fwd tau_strafe", source=["p.tau[FWD]", "p.tau[STRAFE]"])
a_max = m.vec(
    "accel_max",
    "accel_fwd accel_strafe",
    source=["p.accel_max[FWD]", "p.accel_max[STRAFE]"],
)

px, py, vx, vy = x
ux, uy = u
tau_f, tau_s = tau
a_f, a_s = a_max

# ── Model ───────────────────────────────────────────────────────────
R = m.eq(
    "R",
    sp.Matrix(
        [[sp.cos(heading), -sp.sin(heading)], [sp.sin(heading), sp.cos(heading)]]
    ),
    label="R(θ)",
)
v_body = m.eq("v_body", R.T @ sp.Matrix([vx, vy]))
u_body = m.eq("u_body", R.T @ sp.Matrix([ux, uy]))
a_body = m.eq(
    "a_body",
    sp.Matrix(
        [
            a_f * sp.tanh((u_body[0] - v_body[0]) / (tau_f * a_f)),
            a_s * sp.tanh((u_body[1] - v_body[1]) / (tau_s * a_s)),
        ]
    ),
)
xdot = m.eq("xdot", sp.Matrix([vx, vy, *(R @ a_body)]))
x_next = m.eq("x_next", x + dt * xdot)

# ── Exports and functions ───────────────────────────────────────────
m.export("x_next", x_next, rust_type="State")
m.export("fx", m.jac(x_next, x), rust_type="StateJac")
m.export("fu", m.jac(x_next, u), rust_type="ControlJac")

m.function("step", returns=["x_next"])
m.function("step_with_jacobians", returns=["x_next", "fx", "fu"])
