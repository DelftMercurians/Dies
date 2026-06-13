"""Robot dynamics — translational first-order velocity lag + heading lag.

State `x = [px, py, vx, vy, theta]` global frame.
Control `u = [vx_cmd, vy_cmd, theta_cmd]` global frame.

Heading is now part of the optimised state. The onboard IMU yaw loop (which
tracks a commanded global heading setpoint) is modelled as a first-order lag of
`theta` toward `theta_cmd`, saturated at `omega_max` via tanh — exactly mirroring
the per-axis translational velocity lag. Because `theta` enters the body↔global
rotation `R(theta)`, the planner can now rotate the robot to exploit the
anisotropic translational acceleration limits. Forward-Euler at MPC stage dt.
"""

import sympy as sp

from ilqr_codegen import Model

m = Model("dynamics", title=__doc__)

# ── Rust function args ──────────────────────────────────────────────
m.arg("x", "&State")
m.arg("u", "&Control")
m.arg("dt", "f64")
m.arg("p", "&RobotParams")

# ── Symbol bindings (sym → Rust scalar source) ──────────────────────
x = m.vec("x", "px py vx vy theta", source="x[{i}]")
u = m.vec("u", "ux uy theta_cmd", source="u[{i}]")
dt = m.scalar("dt")
tau = m.vec("tau", "tau_fwd tau_strafe", source=["p.tau[FWD]", "p.tau[STRAFE]"])
a_max = m.vec(
    "accel_max",
    "accel_fwd accel_strafe",
    source=["p.accel_max[FWD]", "p.accel_max[STRAFE]"],
)
tau_yaw = m.scalar("tau_yaw", "p.tau_yaw")
omega_max = m.scalar("omega_max", "p.omega_max")

px, py, vx, vy, theta = x
ux, uy, theta_cmd = u
tau_f, tau_s = tau
a_f, a_s = a_max

# ── Model ───────────────────────────────────────────────────────────
R = m.eq(
    "R",
    sp.Matrix([[sp.cos(theta), -sp.sin(theta)], [sp.sin(theta), sp.cos(theta)]]),
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
theta_dot = m.eq(
    "theta_dot",
    omega_max * sp.tanh((theta_cmd - theta) / (tau_yaw * omega_max)),
)
xdot = m.eq("xdot", sp.Matrix([vx, vy, *(R @ a_body), theta_dot]))
x_next = m.eq("x_next", x + dt * xdot)

# ── Exports and functions ───────────────────────────────────────────
m.export("x_next", x_next, rust_type="State")
m.export("fx", m.jac(x_next, x), rust_type="StateJac")
m.export("fu", m.jac(x_next, u), rust_type="ControlJac")

m.function("step", returns=["x_next"])
m.function("step_with_jacobians", returns=["x_next", "fx", "fu"])
