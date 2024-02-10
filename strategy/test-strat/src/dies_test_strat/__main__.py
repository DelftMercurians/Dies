from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control

print("Starting test-strat")

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
    u = [0.2, 0]
    tolerance = 0.01

    to_save = []
    start_time = None
    last_time = time.time()
    try:
        while True:
            msg = bridge.recv()
            if not msg:
                continue

            print([*msg.own_players, *msg.opp_players])
            continue
            player_id = 12
            if len(msg.own_players) == 0:
                print("No own players not found")
                continue
            player = next((p for p in msg.own_players if p.id == player_id), None)
            if player is None:
                print("Player not found")
                continue
            print(f"Player position: {player.position}")
            continue

            bridge.send(PlayerCmd(player_id, *u))
            to_save.append(
                {
                    "time": time.time(),
                    "position": np.array(player.position),
                    "velocity": np.array(player.velocity),
                    "orientation": player.orientation,
                    "u_target": u,
                }
            )
            last_time = time.time()

            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 20:
                print("Time limit reached")
                bridge.send(PlayerCmd(player_id, 0, 0))
                break

            if (
                player.position[0] < pos_constraints[0] * 1000
                or player.position[0] > pos_constraints[1] * 1000
                or player.position[1] < pos_constraints[2] * 1000
                or player.position[1] > pos_constraints[3] * 1000
            ):
                print("Out of bounds")
                bridge.send(PlayerCmd(player_id, 0, 0))
                continue

            print(f"velocity: {player.velocity}")
            if (
                sqrt(player.velocity[0] ** 2 + player.velocity[1] ** 2)
                > sqrt(u[0] ** 2 + u[1] ** 2) + 0.03
            ):
                print("Too fast")
                bridge.send(PlayerCmd(player_id, 0, 0))
                continue
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(e)
    finally:
        print("Exiting")
        # Find all file saved as traj##.npy
        idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
        last_idx = max(idxs) if len(idxs) > 0 else 0
        np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
        print("Saved trajectory")
