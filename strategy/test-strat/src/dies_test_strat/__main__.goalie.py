from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd
import snoop

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

print("Starting test-strat")

f = KalmanFilter(dim_x=4, dim_z=2)
f.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])

R1 = 10
f.P *= np.array([[R1, 0, 0, 0], [0, R1, 0, 0], [0, 0, R1, 0], [0, 0, 0, R1]])
f.R = np.array([[R1, 0], [0, R1]])


pos_bounds = [
    -3000,
    3000,
    -3000,
    3000,
]  # Position Constraints [mm]:    [x_min, x_max, y_min, y_max]


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


class PID:
    def __init__(self, dim, Kp, Kd=0.0, Ki=0.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dim = dim
        self.integral = np.zeros(dim)
        self.prev_inp = np.zeros(dim)
        self.target = None

    def set_target(self, target):
        # self.integral = np.zeros(self.dim)
        self.target = np.array(target)

    def step(self, current, error=None):
        assert self.target is not None, "No target!"
        if error is None:
            error = np.array(self.target) - np.array(current)
        else:
            error = np.array(error)
        self.integral += error
        derivative = current - self.prev_inp
        self.prev_inp = current
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def get_params(self):
        return np.array([self.Kp, self.Ki, self.Kd])


if __name__ == "__main__":
    bridge = Bridge()

    player_id = 14
    rid = 2

    # heading_pid = PID(dim=1, Kp=0.25, Ki=0.015, Kd=2.6)
    heading_Kp_base = 1.2
    heading_pid = PID(dim=1, Kp=heading_Kp_base, Ki=0.05, Kd=2)
    heading_pid.set_target(0)

    pos_pid = PID(dim=2, Kp=0.7, Ki=0.01, Kd=0.6)
    pos_pid.set_target([0, 0])

    targets = np.array(
        [
            [-700, -700],
            [-700, 700],
            # [700, 700],
            # [-700, 700],
        ]
    )
    target_idx = 0

    to_save = []
    start_time = None
    last_time = time.time()
    f_init = False
    w = 0
    done_facing = False
    ball_n = 4
    ball_pos_avg = np.zeros((ball_n, 2))
    idx = 0
    u = None
    try:
        while True:
            msg = bridge.recv()
            if not msg:
                continue

            if len(msg.own_players) == 0:
                print("No own players not found")
                continue
            # print(f"Own players: {msg.own_players}")
            player = next((p for p in msg.own_players if p.id == player_id), None)
            if player is None:
                print("Player not found")
                continue
            if msg.ball is None:
                print("Ball not found")
                continue

            if start_time is None:
                start_time = time.time()

            if not f_init:
                f.x = np.array([player.position[0], player.position[1], 0, 0])
                f_init = True

            ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
            ball_pos_avg[idx] = ball_pos
            ball_pos = ball_pos_avg.mean(axis=0)
            idx = (idx + 1) % ball_n
            pos = np.array([player.position[0], player.position[1]])
            phi = player.orientation
            target = targets[target_idx]

            dt = time.time() - last_time
            f.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            f.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.013)

            f.predict()
            f.update(pos)
            # vel = (f.x[0:2] - prev_x[0:2]) / dt
            vel = f.x[2:4]

            # Face the ball
            # heading = np.arctan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0])
            # Face forward
            # heading = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            # heading_pid.set_target(heading)
            # The error in angle between phi and the target heading (-pi, pi)
            err = heading_pid.target - phi
            err += 2 * pi if err < -pi else 0
            err -= 2 * pi if err > pi else 0
            if abs(err) < 0.05 and not done_facing:
                start_time = time.time()
                done_facing = True

            target = np.array([-700, ball_pos[1]])
            pos_pid.set_target(target)
            # dist = np.linalg.norm(pos - target)
            # if dist < 200:
            #     target_idx = (target_idx + 1) % len(targets)
            #     done_facing = False
            #     # start_time = None
            #     print("Reached target")
            #     continue

            w = float(heading_pid.step(phi, err)[0])
            if done_facing and (time.time() - start_time) > 0.4:
                u = pos_pid.step(pos) / 1000
                vx, vy = global_to_local_vel(float(u[0]), float(u[1]), phi + w * dt)
                # heading_pid.Kp = min(vx * abs(vx) * 0.02, heading_Kp_base)
                bridge.send(PlayerCmd(rid, vx, -vy, w))
            else:
                bridge.send(PlayerCmd(rid, 0, 0, w))

            to_save.append(
                {
                    "time": time.time(),
                    "position": pos,
                    "f_position": f.x[0:2],
                    "velocity": vel,
                    "orientation": phi,
                    "w": w,
                    "u": u,
                    "heading_pid": heading_pid.get_params(),
                    "pos_pid": pos_pid.get_params(),
                    # "ball_pos": ball_pos,
                }
            )
            last_time = time.time()

            # if time.time() - start_time > 5:
            #     print("Time limit reached")
            #     bridge.send(PlayerCmd(rid, 0, 0))
            #     break

            to_sleep = (1 / 20) - dt
            if to_sleep > 0:
                time.sleep(to_sleep)

            # if (
            #     pos[0] < pos_bounds[0] * 1000
            #     or pos[0] > pos_bounds[1] * 1000
            #     or pos[1] < pos_bounds[2] * 1000
            #     or pos[1] > pos_bounds[3] * 1000
            # ):
            #     print("Out of bounds, position: ", pos)
            #     bridge.send(PlayerCmd(rid, 0, 0))
            #     continue

            # if np.linalg.norm(vel) > 10:
            #     print("Too fast, velocity: ", vel)
            #     bridge.send(PlayerCmd(rid, 0, 0))
            #     continue
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        print(e)
        print("Stopping")

    bridge.send(PlayerCmd(rid, 0, 0))
    print("Exiting Python")
    # Find all file saved as traj##.npy
    idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
    last_idx = max(idxs) if len(idxs) > 0 else 0
    np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
    print("Saved trajectory")
