from math import e
import sys
import numpy as np
import casadi as ca

num_states = 2
num_inputs = 2


def mpc_control(
    N,
    x_init,
    u_init,
    x_target,
    pos_constraints,
    vel_constraints,
    # acc_constraints,
    obstacles,
    move_obstacles,
    dt,
    last_plan,
    tw=0,
):
    assert tw // dt >= 0
    assert tw // dt < N

    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    Q = np.eye(2) * 0.4
    R = np.eye(2) * 0.6

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Init Constraint
    constraints += [x[:, 0] == x_init]
    # constraints += [u[:, 0] == u_init]

    # Loop through time steps
    for k in range(N):
        # State Constraints
        constraints += [x[:, k] >= [pos_constraints[0], pos_constraints[2]]]
        constraints += [x[:, k] <= [pos_constraints[1], pos_constraints[3]]]
        # constraints += [u[:, k] >= [vel_constraints[0], vel_constraints[2]]]
        # constraints += [u[:, k] <= [vel_constraints[1], vel_constraints[3]]]

        for obstacle in obstacles:
            euclid_distance = ca.norm_2(
                x[0:2, k] - np.array(obstacle[0:2]).reshape(2, 1)
            )
            constraints += [euclid_distance >= obstacle[2] + 100]
            # cost += 100/((euclid_distance-obstacle[2])**2 + 0.01)

        # Dynamics Constraint
        constraints += [
            x[:, k + 1]
            == (
                (x[:, k] + dt * u[:, k - (tw // dt)])
                if k - (tw // dt) >= 0
                else x[:, k]
            )
        ]

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N) + ca.mtimes(u[:, -1].T, R @ u[:, -1])

    # Warm start
    if last_plan is not None:
        opti.set_initial(x, last_plan)
    else:
        # Setup initial guess for states (interpolate between x_init and x_target)
        for k in range(N + 1):
            x_guess = x_init + (x_target - x_init) * k * dt * 0.04
            opti.set_initial(x[:, k], x_guess)

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
    }  # Set the verbosity level (0-12, default is 5)
    opti.solver(
        "ipopt",
        opts,
        opts,
    )

    # Run Solver
    try:
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        # optimal_cost = sol.value(cost)
    except RuntimeError:
        print("Solver failed to find a solution.")
        opti.debug.x_describe(0)
        opti.debug.g_describe(0)

        # opti.debug.show_infeasibilities()
        x = opti.debug.value(x)
        u = opti.debug.value(u)
        # return u[:, 0], x[:, 1], x
        raise

    return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
