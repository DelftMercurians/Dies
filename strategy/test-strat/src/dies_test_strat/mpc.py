from math import e
import sys
import numpy as np
import casadi as ca


def mpc_control(
    vehicle,
    N,
    x_init,
    u_init,
    x_target,
    pos_constraints,
    vel_constraints,
    # acc_constraints,
    obstacles,
    move_obstacles,
    last_plan,
    max_vel=1,
    v_max_penalty=100,
    num_states=2,
    num_inputs=2,
):
    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    Q = np.eye(2) * 0.04
    R = np.eye(2) * 0.06

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Init Constraint
    opti.subject_to(x[:, 0] == x_init)
    # opti.subject_to(u[:, 0] == u_init)

    # Loop through time steps
    for k in range(N):
        # State Constraints
        opti.subject_to(x[:, k] >= [pos_constraints[0], pos_constraints[2]])
        opti.subject_to(x[:, k] <= [pos_constraints[1], pos_constraints[3]])
        opti.subject_to(u[:, k] >= [vel_constraints[0], vel_constraints[2]])
        opti.subject_to(u[:, k] <= [vel_constraints[1], vel_constraints[3]])

        # Acceleration constraint
        # TODO

        # for obstacle in move_obstacles:
        #     euclid_distance = ca.norm_2(
        #         x[0:2, k] - np.array(obstacle[0:2]).reshape(2, 1)
        #     )
        #     opti.subject_to(euclid_distance >= obstacle[2] + 0.003)
        # cost += 100/((euclid_distance-obstacle[2])**2 + 0.01)

        for obstacle in obstacles:
            euclid_distance = ca.norm_2(
                (x[0:2, k] - np.array(obstacle[0:2]).reshape(2, 1))
                + np.ones((2, 1)) * sys.float_info.epsilon
            )
            # opti.subject_to(euclid_distance >= obstacle[2])
            cost += 100 / ((euclid_distance - obstacle[2]) ** 2 + 0.01)

        # Dynamics Constraint
        opti.subject_to(x[:, k + 1] == vehicle.A @ x[:, k] + vehicle.B @ u[:, k])

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])

        # Veclocity cost
        # vel = u[0, k] + u[1, k]
        # v_excess = vel - max_vel
        # cost += v_max_penalty * ca.if_else(v_excess > 0, v_excess, 0)

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N) + ca.mtimes(u[:, -1].T, R @ u[:, -1])

    # Warm start
    if last_plan is not None:
        opti.set_initial(x, last_plan)
    else:
        # Setup initial guess for states (interpolate between x_init and x_target)
        for k in range(N + 1):
            x_guess = x_init + (x_target - x_init) * k / N
            # x_guess[6] = 0  # Set initial guess for speed in z direction to zero
            opti.set_initial(x[:, k], x_guess)

    # Define Problem in solver
    opti.minimize(cost)

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
        # opti.callback(lambda i: print(opti.debug.value(x)))
        sol = opti.solve()
        optimal_solution_u = sol.value(u)
        optimal_solution_x = sol.value(x)
        optimal_cost = sol.value(cost)
        # print("Optimal cost:", optimal_cost)
    except RuntimeError:
        print("Solver failed to find a solution.")
        opti.debug.show_infeasibilities()
        # opti.debug.x_describe()
        print()
        raise

    return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
