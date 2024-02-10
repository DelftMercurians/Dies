from math import sqrt, cos, sin, pi
import time
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control

print("Starting test-strat")

# MPC parameters
dt = 1 / 30  # Time step [s]
N = 10  # Time Horizon

acc_constraints = [
    -1,
    1,
    -1,
    1,
]  # Acceleration Constraints [m/s^2]:    [x_min, x_max, y_min, y_max]
vehicle = vehicle_SS(dt)

pos_constraints = [
    -2,
    2,
    -2.5,
    2.5,
]  # Position Constraints [m]:    [x_min, x_max, y_min, y_max]


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


if __name__ == "__main__":
    bridge = Bridge()
    x_target = [0, -800]
    tolerance = 0.01

    to_save = []
    start_time = None
    last_time = time.time()
    try:
        while True:
            msg = bridge.recv()
            if not msg:
                continue

            player_id = 12
            if len(msg.own_players) == 0:
                print("No own players not found")
                continue
            player = next((p for p in msg.own_players if p.id == player_id), None)
            if player is None:
                print("Player not found")
                continue

            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 20:
                print("Time limit reached")
                bridge.send(PlayerCmd(player_id, 0, 0))
                break

            other_players = [
                p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id
            ]

            dist = sqrt(
                (player.position[0] - x_target[0]) ** 2
                + (player.position[1] - x_target[1]) ** 2
            )
            if dist < tolerance:
                print("Reached target")
                break
            else:
                x_init = [
                    player.position[0] / 1000,
                    player.position[1] / 1000,
                ]

                if (
                    x_init[0] < pos_constraints[0]
                    or x_init[0] > pos_constraints[1]
                    or x_init[1] < pos_constraints[2]
                    or x_init[1] > pos_constraints[3]
                ):
                    print("Out of bounds")
                    bridge.send(PlayerCmd(player_id, 0, 0))
                    continue

                print(f"velocity: {player.velocity}")
                if sqrt(player.velocity[0] ** 2 + player.velocity[1] ** 2) > 1000:
                    print("Too fast")
                    bridge.send(PlayerCmd(player_id, 0, 0))
                    continue

                obstacles = [
                    [p.position[0] / 1000, p.position[1] / 1000, 0.1]
                    for p in other_players
                ]

                print(f"Player position: {x_init}")
                print(f"Obstacles position: {obstacles}")
                u, _, traj = mpc_control(
                    vehicle=vehicle,
                    N=N,
                    x_init=np.array(x_init),
                    u_init=np.array([player.velocity[0], player.velocity[1]]) / 1000,
                    x_target=np.array(x_target) / 1000,
                    pos_constraints=pos_constraints,
                    vel_constraints=[
                        -0.04,
                        0.04,
                        -0.04,
                        0.04,
                    ],  # [v_x_min, v_x_max, v_y_min, v_y_max]
                    max_vel=0.08,
                    v_max_penalty=300,
                    obstacles=obstacles,  # [center_x, center_y, radius]
                    move_obstacles=[],
                    last_plan=None,
                )
                to_save.append(
                    {
                        "position": player.position,
                        "velocity": player.velocity,
                        "u": u,
                        "obstacles": obstacles,
                        "traj": traj,
                    }
                )

                u_x = u[0] * 1000
                u_y = u[1] * 1000
                u_x, u_y = global_to_local_vel(u_x, u_y, player.orientation)
                curr_time = time.time()
                last_time = curr_time
                print(f"Sending command: {u_x}, {u_y}")
                bridge.send(PlayerCmd(player_id, int(u_x), int(u_y)))
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(e)
    finally:
        print("Exiting")
        np.save("traj.npy", to_save)
        print("Saved trajectory")
