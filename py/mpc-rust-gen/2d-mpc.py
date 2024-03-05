import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np

# Build parametric optimizer
# ------------------------------------


# (nu, nx, N, L, ts) = (1, 2, 20, 0.5, 0.1)
# (posref, velref) = (1, 0)
# (q, qtheta, r, qN, qthetaN) = (10, 0.1, 1, 200, 2)

# u = cs.SX.sym('u', nu*N)
# x = cs.SX.sym('x', nx*(N+1))

# posref = 1.0

# # (pos, vel) = (z0[0], z0[1])

# cost = 0
# for t in range(0, nu*N, nu):
#     pos_t = x[2*t]
#     vel_t = x[2*t +1]
#     u_t = u[t]
#     cost += q*((pos_t-posref)**2 + (u_t)**2)
  
#     pos_t = x[2*(t+1)]
#     vel_t = x[2*(t+1) +1]

# cost += qN*((pos_t-posref)**2)

# ux, uy are velocities
(nu, nx, N, L, ts) = (2, 2, 20, 0.5, 0.1)
(q, qtheta, r, qN, qthetaN) = (10, 0.1, 1, 200, 2)

u = cs.SX.sym('u', nu*N)
p = cs.SX.sym('p', 2*nx)

(x, y) = (p[0], p[1])
(xref, yref) = (p[2], p[3])

cost = 0
# c = 0
for t in range(0, nu*N, nu):

    ux_t = u[t]
    uy_t = u[t+1]

    cost += q*((x-xref)**2 + (y-yref)**2 + (ux_t)**2 + (uy_t)**2)
  
    x += ts * ux_t
    y += ts * uy_t

    # avoid obs at origin
    # c += cs.fmax(0, 1 - x**2 - y**2)

cost += qN*((x-xref)**2 + (y-yref)**2)

umin = [-5.0] * (nu*N)
umax = [5.0] * (nu*N)
bounds = og.constraints.Rectangle(umin, umax)

problem = og.builder.Problem(u, p, cost).with_constraints(bounds)
build_config = og.config.BuildConfiguration()\
    .with_build_directory("my_optimizers")\
    .with_build_mode("debug")\
    .with_build_python_bindings()
meta = og.config.OptimizerMeta()\
    .with_optimizer_name("navigation")
solver_config = og.config.SolverConfiguration()\
    .with_tolerance(1e-7)
builder = og.builder.OpEnOptimizerBuilder(problem,
                                          meta,
                                          build_config,
                                          solver_config)
builder.build()



# # Use TCP server
# # ------------------------------------
# mng = og.tcp.OptimizerTcpManager('my_optimizers/navigation')
# mng.start()

# mng.ping()
# solution = mng.call([-1.0, 2.0, 0.0], initial_guess=[1.0] * (nu*N))
# mng.kill()


# # Plot solution
# # ------------------------------------
# time = np.arange(0, ts*N, ts)
# u_star = solution['solution']
# ux = u_star[0:nu*N:2]
# uy = u_star[1:nu*N:2]

# plt.subplot(211)
# plt.plot(time, ux, '-o')
# plt.ylabel('u_x')
# plt.subplot(212)
# plt.plot(time, uy, '-o')
# plt.ylabel('u_y')
# plt.xlabel('Time')
# plt.show()

# Note: to use the direct interface you need to build using
#       .with_build_python_bindings()
import sys

# Use Direct Interface
# ------------------------------------
sys.path.insert(1, './my_optimizers/navigation')
import navigation

solver = navigation.solver()
x_init = [-1., -1.]
x_ref = [2., 2.]
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
plt.plot(xx, xy, '-o')
plt.show()
