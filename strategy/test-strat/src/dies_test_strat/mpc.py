import numpy as np
import casadi as ca


def mpc_control(
    vehicle,
    N,
    x_init,
    x_target,
    pos_constraints,
    max_vel,
    acc_constraints,
    obstacles,
    move_obstacles,
    initial_guess_x,
    v_max_penalty=1,
    num_states=4,
    num_inputs=2,
):
    # Create an optimization problem
    opti = ca.Opti()

    # State & Input matrix
    Q = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    R = np.eye(2)

    # Define Variables
    x = opti.variable(num_states, N + 1)
    u = opti.variable(num_inputs, N)

    # Initialize Cost & Constraints
    cost = 0.0
    constraints = []

    # Loop through time steps
    for k in range(N):
        # State Constraints
        constraints += [
            x[:2, k]
            >= [
                pos_constraints[0],
                pos_constraints[2],
            ]
        ]
        constraints += [
            x[:2, k]
            <= [
                pos_constraints[1],
                pos_constraints[3],
            ]
        ]
        # constraints += [u[:, k] >= [acc_constraints[0], acc_constraints[2]]]
        # constraints += [u[:, k] <= [acc_constraints[1], acc_constraints[3]]]

        for obstacle in obstacles:
            euclid_distance = ca.norm_2(
                x[0:2, k] - np.array(obstacle[0:2]).reshape(2, 1)
            )
            constraints += [euclid_distance >= obstacle[2]]
            # cost += 1000/((euclid_distance-obstacle[2])**2 + 0.01)

        for obstacle in move_obstacles:
            euclid_distance = ca.norm_2(
                x[0:2, k] - np.array(obstacle[0:2]).reshape(2, 1)
            )
            constraints += [euclid_distance > obstacle[2]]
            cost += 100 / ((euclid_distance - obstacle[2]) ** 2 + 0.01)

        # Dynamics Constraint
        constraints += [x[:, k + 1] == vehicle.A @ x[:, k] + vehicle.B @ u[:, k]]

        # Cost function
        e_k = x_target - x[:, k]
        cost += ca.mtimes(e_k.T, Q @ e_k) + ca.mtimes(u[:, k].T, R @ u[:, k])

        # Veclocity cost
        vel = x[2, k] + x[3, k]
        v_excess = vel - max_vel
        cost += v_max_penalty * ca.if_else(v_excess > 0, v_excess, 0)

    # Init Constraint
    constraints += [x[:, 0] == x_init]

    # Cost last state
    e_N = x_target - x[:, -1]
    cost += ca.mtimes(e_N.T, Q @ e_N)

    # Warm start initialization if initial guesses are provided
    # print(f"        Initial Guess: {initial_guess_x}")
    if initial_guess_x is not None:
        opti.set_initial(x, initial_guess_x)

    # Define Problem in solver
    opti.minimize(cost)
    opti.subject_to(constraints)

    opti.solver(
        "ipopt",
        {"ipopt.print_level": 0, "print_time": 0},
        {"ipopt.print_level": 0, "print_time": 0},
    )

    # Run Solver
    sol = opti.solve()
    optimal_solution_u = sol.value(u)
    optimal_solution_x = sol.value(x)
    # optimal_cost = sol.value(cost)
    # print("Optimal cost:", optimal_cost)

    return optimal_solution_u[:, 0], optimal_solution_x[:, 1], optimal_solution_x
