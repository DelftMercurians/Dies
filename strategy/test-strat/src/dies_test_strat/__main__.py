from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term

print("Starting test-strat")


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


class SavGolFilter:
    """Uses a savitky-golay filter to compute velocity from position"""

    def __init__(self, dim, window=7) -> None:
        self.window = window
        self.state = np.zeros((self.window, dim))

    def update(self, x):
        from scipy.signal import savgol_filter

        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = np.array(x)
        dx = savgol_filter(self.state, self.window, 2, deriv=1, axis=0)

        return dx[-1]


class PID:
    def __init__(self, dim, Kp, Kd=0.0, Ki=0.0, diff_func=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dim = dim
        self.integral = np.zeros(dim)
        self.prev_inp = np.zeros(dim)
        self.target = None
        self.diff_func = diff_func if diff_func is not None else lambda a, b: a - b

    def set_target(self, target):
        self.target = np.array(target)

    def step(self, current, error=None, derivative=None):
        assert self.target is not None, "No target!"
        if error is None:
            error = self.diff_func(np.array(self.target), np.array(current))
        else:
            error = np.array(error)
        self.integral += error
        if derivative is None:
            derivative = self.diff_func(current, self.prev_inp)
        else:
            derivative = np.array(derivative)
        self.prev_inp = current
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def get_params(self):
        return np.array([self.Kp, self.Ki, self.Kd])


def angle_diff(target, phi):
    return (target - phi + np.pi) % (2 * np.pi) - np.pi


if __name__ == "__main__":
    bridge = Bridge()

    player_id = 14
    rid = 2

    v_filter = SavGolFilter(dim=2)
    w_filter = SavGolFilter(dim=1)

    heading_Kp_base = 1.2
    heading_pid = PID(dim=1, Kp=heading_Kp_base, Ki=0.01, Kd=2.4, diff_func=angle_diff)
    heading_pid.set_target(pi)

    pos_pid = PID(dim=2, Kp=2, Ki=0.00, Kd=0.0)
    pos_pid.set_target([0, 0])

    targets = np.array(
        [
            [-500, -700],
            [-500, 700],
        ]
    )
    target_idx = 0

    to_save = []
    start_time = None
    last_time = time.time()
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
            if isinstance(msg, Term):
                break

            if len(msg.own_players) == 0:
                print("No own players not found")
                continue
            player = next((p for p in msg.own_players if p.id == player_id), None)
            if player is None:
                print("Player not found")
                continue
            if msg.ball is None:
                print("Ball not found")
                continue

            if start_time is None:
                start_time = time.time()

            # other_players = [
            #     p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id
            # ]
            # if len(other_players) == 0:
            #     print("No other players found")
            #     continue

            ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
            idx = (idx + 1) % ball_n
            ball_pos_avg[idx] = ball_pos
            ball_pos = ball_pos_avg.mean(axis=0)
            pos = np.array([player.position[0], player.position[1]])
            vel = v_filter.update(pos)
            phi = player.orientation
            ang_vel = w_filter.update(phi)
            dt = time.time() - last_time
            last_time = time.time()
            print("Vel", vel)

            target = targets[target_idx]
            dist = np.linalg.norm(pos - target)
            pos_pid.set_target(target)
            u = pos_pid.step(pos)
            if dist < 50:
                target_idx = (target_idx + 1) % len(targets)
                done_facing = False
                print("Reached target")
                continue

            # Face forward (+x axis)
            heading_pid.set_target(0)
            # The error in angle between phi and the target heading (-pi, pi)
            err = angle_diff(heading_pid.target, phi)
            if abs(err) < 0.1:
                start_time = time.time()
                done_facing = True
            # elif abs(err) > 0.5:
            #     done_facing = False
            #     start_time = None

            if done_facing:
                v = np.linalg.norm(u)
                u /= 1000.0
                vx, vy = global_to_local_vel(float(u[0]), float(u[1]), phi + (w * dt))
                # heading_pid.Kp = 1.5 if v > 0.0 else 1.3
                # #     (3 * v) + heading_Kp_base if v > 0.1 else heading_Kp_base
                # # )
                # heading_pid.Kd = 2.4 if v > 0 else 0
                # heading_pid.Ki = 0.01 if v > 0 else 0
                w = float(heading_pid.step(phi, err, derivative=ang_vel)[0])
                bridge.send(PlayerCmd(rid, vx, -vy, 0))
            else:
                heading_pid.Kd = 0.0
                heading_pid.Ki = 0.0
                w = float(heading_pid.step(phi, err, derivative=ang_vel)[0])
                bridge.send(PlayerCmd(rid, 0, 0, w))

            to_save.append(
                {
                    "time": time.time(),
                    "position": pos,
                    # "f_position": f.x[0:2],
                    "velocity": vel,
                    "orientation": phi,
                    "w": w,
                    "u": u,
                    "heading_pid": heading_pid.get_params(),
                    "pos_pid": pos_pid.get_params(),
                    "phi_err": err,
                    "ball_pos": ball_pos,
                    # "plan": plan,
                }
            )

            to_sleep = (1 / 20) - dt
            if to_sleep > 0:
                time.sleep(to_sleep)

    except (KeyboardInterrupt, SystemExit) as e:
        print(e)
        print("Stopping")
    finally:
        print("Exiting Python")
        # Find all file saved as traj##.npy
        idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
        last_idx = max(idxs) if len(idxs) > 0 else 0
        np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
        print("Saved trajectory")

        bridge.send(PlayerCmd(rid, 0, 0))
