import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import sys

(nu, nx, N, L, ts) = (2, 2, 30, 0.5, 0.1)
(q, qu, r, qN, qthetaN) = (10, 1, 1, 200, 2)
(alpha, Hw) = (0.1, 4*nu)

u = cs.SX.sym('u', nu*N)
p = cs.SX.sym('p', 2*nx)

(x, y) = (p[0], p[1])
(xref, yref) = (p[2], p[3])

cost = 0
c = 0
# Jumping from 0 to 2, 4, 6, 8
for t in range(0, nu*N, nu):

    #This is done to make sure that the last value of u is not used when t < Hw
    if t < Hw:
        ux_t = 0
        uy_t = 0
    else:
        ux_t = u[t - Hw]
        uy_t = u[t+1 - Hw] 
    
    print(f"{ux_t = }", f"{uy_t = }")

    cost += q*((x-xref)**2 + (y-yref)**2) + qu * ((ux_t)**2 + (uy_t)**2)
  
    x += ts * ux_t 
    y += ts * uy_t
    c += cs.fmax(0, 1.5 - x**2 - y**2)

cost += qN*((x-xref)**2 + (y-yref)**2)

umin = [-3.0] * (nu*N)
umax = [3.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

# .with_constraints(circle_constraints)
problem = og.builder.Problem(u, p, cost).with_penalty_constraints(c).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("my_optimizers")\
    .with_build_mode("debug")\
    .with_build_python_bindings()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-4)\
    .with_initial_tolerance(1e-4)\
    .with_max_outer_iterations(5)\
    .with_delta_tolerance(1e-2)\
    .with_penalty_weight_update_factor(10.0)\
    .with_initial_penalty(100.0)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()

# Use Direct Interface
# ------------------------------------
sys.path.insert(1, './my_optimizers/navigation')
import navigation

solver = navigation.solver()



x_init = [-2., -2.]
x_ref = [2, 1.8]
result = solver.run(p=x_init + x_ref,
                    initial_guess=[1.0] * (nu*N))
u_star = result.solution

# Plot solution
# ------------------------------------
time = np.arange(0, ts*N, ts)
ux = u_star[0:nu*N:2]
uy = u_star[1:nu*N:2]

plt.subplot(311)
plt.plot(time, ux, '-o')
plt.ylabel('u_x')
plt.subplot(212)
plt.plot(time, uy, '-o')
plt.ylabel('u_y')
plt.xlabel('Time')
plt.show()

x_states = [0.0] * (nx*(N+2))
x_states[0:nx+1] = x_init
for t in range(0, N):
    u_t = u_star[t*nu:(t+1)*nu]

    x = x_states[t * nx]
    y = x_states[t * nx + 1]

    x_states[(t+1)*nx] = x + ts * (u_t[0])
    x_states[(t+1)*nx+1] = y + ts * (u_t[1])

time = np.arange(0, ts*N, ts)

xx = x_states[0:nx*N:nx]
xy = x_states[1:nx*N:nx]

print(f"{x_states = }")
print(xx)
fig, ax = plt.subplots()
ax.plot(xx, xy, '-o')
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
ax.add_patch(circle)
plt.show()
