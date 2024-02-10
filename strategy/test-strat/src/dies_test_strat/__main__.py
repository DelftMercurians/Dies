from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

print("Starting test-strat")

f = KalmanFilter(dim_x=4, dim_z=2)
f.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
f.P *= 1000.0
f.R = np.array([[5, 0], [0, 5]])
dt = 1 / 30
f.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
f.Q = Q_discrete_white_noise(dim=4, dt=1 / 30, var=0.13)

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
    u = [1, 0]
    tolerance = 0.01

    to_save = []
    start_time = None
    last_time = time.time()
    f_init = False
    try:
        while True:
            msg = bridge.recv()
            if not msg:
                continue

            player_id = 14
            rid = 2
            if len(msg.own_players) == 0:
                print("No own players not found")
                continue
            # print(f"Own players: {msg.own_players}")
            player = next((p for p in msg.own_players if p.id == player_id), None)
            if player is None:
                print("Player not found")
                continue
            # print(f"Player position: {player.position}")
            if not f_init:
                f.x = np.array([player.position[0], player.position[1], 0, 0])
            pos = np.array([player.position[0], player.position[1]])
            phi = player.orientation

            dt = time.time() - last_time

            f.predict()
            f.update(pos)
            vel = f.x[2:4]
            print(f"Velocity: {vel}")

            bridge.send(PlayerCmd(rid, u[0], u[1], 0))
            to_save.append(
                {
                    "time": time.time(),
                    "position": pos,
                    "velocity": vel,
                    "orientation": phi,
                    "u_target": u,
                }
            )
            last_time = time.time()

            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 20:
                print("Time limit reached")
                bridge.send(PlayerCmd(rid, 0, 0))
                break

            # if (
            #     pos[0] < pos_constraints[0] * 1000
            #     or pos[0] > pos_constraints[1] * 1000
            #     or pos[1] < pos_constraints[2] * 1000
            #     or pos[1] > pos_constraints[3] * 1000
            # ):
            #     print("Out of bounds, position: ", pos)
            #     bridge.send(PlayerCmd(rid, 0, 0))
            #     continue

            if np.linalg.norm(vel) > 1:
                print("Too fast, velocity: ", player.velocity)
                bridge.send(PlayerCmd(rid, 0, 0))
                continue
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(e)
    finally:
        print("Exiting")
        if len(to_save) > 2:
            # Find all file saved as traj##.npy
            idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
            last_idx = max(idxs) if len(idxs) > 0 else 0
            np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
            print("Saved trajectory")
