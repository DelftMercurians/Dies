import opengen as og
import casadi.casadi as cs
import matplotlib.pyplot as plt
import numpy as np
import sys

def make_solver(
    N,
    x_init,
    x_target,
    pos_constraints,
    u_constraints, # min and max list
    obstacles,
    move_obstacles,
    ts,
    alpha=1,
    Hw=1,
 ):
    # Define the optimization problem
    # ------------------------------------
    (nu, nx, N, L, ts) = (2, 2, 30, 0.5, 0.1)
    (q, qu, r, qN, qthetaN) = (10, 1, 1, 200, 2)
    (x, y) = x_init[0], x_init[1]

    u = cs.SX.sym('u', nu*N)
    p = cs.SX.sym('p', 2*nx)
    Hw = Hw*nu
    x_ref, y_ref = x_target[0], x_target[1]

    c = 0
    cost = 0
    for t in range(0, nu*N, nu):
        if t < Hw:
            ux_t = 0
            uy_t = 0
        else:
            ux_t = alpha * u[t - Hw]
            uy_t = alpha * u[t+1 - Hw] 
    
        cost += q*((x-x_ref)**2 + (y-y_ref)**2) + qu * ((ux_t)**2 + (uy_t)**2)  # minimize distance to target

        x += ts * ux_t
        y += ts * uy_t
        for obstacle in obstacles:
            c += cs.fmax(0, 1.5 - (x-obstacle[0])**2 - (y-obstacle[1])**2)
    
    cost += qN*((x-x_ref)**2 + (y-y_ref)**2)  # minimize distance to target

    umin = u_constraints[0] * (nu*N)
    umax = u_constraints[1] * (nu*N)
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

    sys.path.insert(1, './my_optimizers/navigation')
    import navigation
    solver = navigation.solver()
    return solver 
        

def mpc_control(solver, x_init, x_ref, obstacles, prev_u):
    result = solver.run(p=x_init + x_ref + obstacles, initial_guess=prev_u)
    return result.solution.u[0], result.solution.u


if __name__ == "__main__":
    Nsim = 100

    N = 60
    x_init = []
    x_ref = [-750, -750]
    obstacles = []
    u_constraints = [-3.0, 3.0]
    ts = 0.02
    prev_u = [1.0] * (2*N)
    solver = make_solver(N, x_init, x_ref, obstacles, u_constraints, ts)

    for t in range(Nsim):
        u_mpc, prev_u = mpc_control(solver, x_init, x_ref, obstacles, prev_u)
    