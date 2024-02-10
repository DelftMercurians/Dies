from mpc import mpc_control
from vehicle import vehicle_SS
import numpy as np

num_states = 2
num_inputs = 2


def simulate(
    simulation_len,
    x_init,
    x_target,
    obstacles,  # [center_x, center_y, radius]
    dt,  # Time step [s]
    N,  # Time Horizon
    pos_constraints,  # Position Constraints [m]:    [x_min, x_max, y_min, y_max]
    vel_constraints,  # Velocity Constraints [m/s]:    [v_x_min, v_x_max, v_y_min, v_y_max]
    acc_constraints,  # Acceleration Constraints [m/s^2]:    [x_min, x_max, y_min, y_max]
    velocity_res,  # m/s
    acceleration,  # m/s^2
):
    vehicle = vehicle_SS(dt)

    ## Timesteps
    timesteps = np.arange(0, simulation_len, dt)
    print(f"Timesteps: {len(timesteps)}")

    # Initialise the output arrays
    x = np.zeros((num_states, len(timesteps) + 1))
    xdot = np.zeros((num_states, len(timesteps) + 1))
    u = np.zeros((num_inputs, len(timesteps)))
    plans = np.zeros((num_states, N + 1, len(timesteps) + 1))

    initial_guess_x = None

    x[:, 0] = np.array(x_init)
    x_target = np.array(x_target)
    for t in range(len(timesteps)):
        # Compute the control input
        u_prev = u[:, max(0, t - 1)]
        u_out, _, plan = mpc_control(
            vehicle=vehicle,
            N=N,
            x_init=x[:, t],
            u_init=u_prev,
            x_target=x_target,
            pos_constraints=pos_constraints,
            vel_constraints=vel_constraints,
            max_vel=0.08,
            v_max_penalty=300,
            obstacles=obstacles,
            move_obstacles=[],
            last_plan=initial_guess_x,
        )
        u[:, t] = u_out
        plans[:, :, t] = plan
        initial_guess_x = plan

        # Apply the control input to the system
        u_diff = u_out - xdot[:, t]
        xdot[:, t + 1] = (
            xdot[:, t] + np.sign(u_diff) * acceleration * dt
            if np.linalg.norm(u_diff) > velocity_res
            else xdot[:, t]
        )
        x[:, t + 1] = xdot[:, t + 1] * dt + x[:, t]

    return x, xdot, u, plans, timesteps, obstacles
