from math import sqrt, cos, sin, pi
import time
from dies_py import Bridge
from dies_py.messages import PlayerCmd

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control

print("Starting test-strat")

# MPC parameters
dt = 1 / 30  # Time step [s]
N = 20  # Time Horizon

acc_constraints = [
    -1,
    1,
    -1,
    1,
]  # Acceleration Constraints [m/s^2]:    [x_min, x_max, y_min, y_max]
vehicle = vehicle_SS(dt)

pos_constraints = [
    -1,
    1,
    -2,
    2,
]  # Position Constraints [m]:    [x_min, x_max, y_min, y_max]


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


if __name__ == "__main__":
    bridge = Bridge()
    x_target = [0, 0, 0, 0]
    tolerance = 0.01

    while True:
        msg = bridge.recv()
        if not msg:
            continue

        if len(msg.own_players) == 0:
            print("No players not found")
            continue
        player = next((p for p in msg.own_players if p.id == 12), None)
        if player is None:
            print("Player not found")
            continue
        rid = player.id

        bridge.send(PlayerCmd(rid, 100, 0, 0))
        dist = sqrt(player.position[0] ** 2 + player.position[1] ** 2)
        if dist < tolerance:
            print("Reached target")
            continue
        else:
            x_init = [
                player.position[0] / 1000,
                player.position[1] / 1000,
                player.velocity[0] / 1000,
                player.velocity[1] / 1000,
            ]
            print(f"x_init {x_init}")
            print(f"x_target {x_target}")

            u, _, __ = mpc_control(
                vehicle=vehicle,
                N=N,
                x_init=x_init,
                x_target=x_target,
                pos_constraints=pos_constraints,
                max_vel=0.08,
                acc_constraints=acc_constraints,
                obstacles=[],  # [center_x, center_y, radius]
                move_obstacles=[],
                initial_guess_x=None,
                num_states=4,
                num_inputs=2,
            )
            u_x = u[0] * 100
            u_y = u[1] * 100
            # print(f"x: {player.position[0]}, y: {player.position[1]}")
            print(f"u_x: {u_x}, u_y: {u_y}")
            u_x, u_y = global_to_local_vel(u_x, u_y, player.orientation)
            print(f"u_x: {u_x}, u_y: {u_y}")
