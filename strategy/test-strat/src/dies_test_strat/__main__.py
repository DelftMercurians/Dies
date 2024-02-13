from math import sqrt, cos, sin, pi
import time
from glob import glob
from dies_test_strat.PRMController import PRMController
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term
import matplotlib.pyplot as plt

print("Starting test-strat")

ROBOT_RADIUS = 90  # mm

pos_bounds = [
    -1400,
    1400,
    -1200,
    1200,
]  # Position Constraints [mm]:    [x_min, x_max, y_min, y_max]


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


def angle_diff(target, phi):
    return (target - phi + np.pi) % (2 * np.pi) - np.pi


class VelFilter:
    """Uses a savitky-golay filter to compute velocity from position"""

    def __init__(self, dim, window=7, alpha=0.6) -> None:
        self.window = window
        self.state = np.zeros((self.window, dim))
        self.dx = np.zeros(dim)
        self.alpha = alpha

    def update(self, x, dt=(1 / 20)):
        from scipy.signal import savgol_filter

        self.state = np.roll(self.state, -1, axis=0)
        self.state[-1] = np.array(x)
        dx = savgol_filter(self.state, self.window, 2, deriv=1, axis=0, delta=dt)
        self.dx = self.dx * self.alpha + (1 - self.alpha) * dx[-1]

        return dx[-1]


class AngVelFilter:
    def __init__(self) -> None:
        self.prev = None
        self.dx = 0.0

    def update(self, x, dt=(1 / 20)):
        assert self.prev is not None
        diff = x - self.prev
        self.prev = x
        if diff > pi:
            diff -= 2 * pi
        elif diff < -pi:
            diff += 2 * pi
        self.dx = self.dx * 0.6 + 0.4 * (diff / dt)
        return self.dx


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


if __name__ == "__main__":
    bridge = Bridge()

    player_id = 14

    dest = np.array([800, 800])
    path = None
    path_idx = 0

    ball_v_filter = VelFilter(dim=2)
    v_filter = VelFilter(dim=2, window=30)
    w_filter = AngVelFilter()

    heading_pid = PID(dim=1, Kp=2.1, Ki=0.0, Kd=0, diff_func=angle_diff)
    heading_pid.set_target(pi)

    pos_pid = PID(dim=2, Kp=0.6, Ki=0.0, Kd=0.0)
    pos_pid.set_target([0, 0])

    to_save = []
    last_time = time.time()
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
            # if msg.ball is None:
            #     print("Ball not found")
            #     continue
            pos = np.array([player.position[0], player.position[1]])

            other_players = [
                p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id
            ]
            obstacles = [
                [p.position[0], p.position[1], ROBOT_RADIUS] for p in other_players
            ]

            if path is None:
                print(f"Creating PRM, detected obstacles: {len(obstacles)}")
                prm = PRMController(
                    numOfRandomCoordinates=500,
                    allObs=obstacles,
                    current=pos,
                    destination=dest,
                    fieldHalfSize=900,
                )
                path = np.array(prm.runPRM(12))
                path = path[np.linalg.norm(path - pos, axis=1) > 200]
                print(pos)
                print(f"Path created: {path}")
                last_time = time.time()
                continue
                # print(f"Start {pos}, Destination {dest}")
                # print(path)
                # plt.scatter(pos[0], pos[1], c="g")
                # plt.scatter(dest[0], dest[1], c="r")
                # for obs in obstacles:
                #     circle = plt.Circle((obs[0], obs[1]), obs[2], color="b", fill=False)
                #     plt.gcf().gca().add_artist(circle)
                # plt.plot(path[:, 0], path[:, 1], "r-")
                # plt.show()
                # break

            dt = max(time.time() - last_time, 1e-6)
            last_time = time.time()
            # ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
            # ball_v = ball_v_filter.update(ball_pos, dt)
            # ball_pred = ball_pos + ball_v * 0.2

            vel = v_filter.update(pos)
            pos_pred = pos + vel * (1 / 20)

            phi = player.orientation
            if w_filter.prev is None:
                w_filter.prev = phi
            ang_vel = w_filter.update(phi)
            # predict what the angle will be in 0.2 seconds
            phi_pred = phi + 0.25 * ang_vel
            if phi_pred > pi:
                phi_pred -= 2 * pi
            elif phi_pred < -pi:
                phi_pred += 2 * pi

            target = path[path_idx]
            dist = np.linalg.norm(target - pos)
            if dist < 90:
                print(f"Reached target {path_idx}")
                path_idx += 1
                if path_idx >= len(path):
                    print("Reached destination")
                    break
                target = path[path_idx]

            pos_pid.set_target(target)
            u = pos_pid.step(pos)

            # Add a repulsive force for each obstacle
            # for obs in obstacles:
            #     dist = np.linalg.norm(obs[:2] - pos)
            #     if dist < obs[2] + 150:
            #         v = pos - obs[:2]
            #         v /= np.linalg.norm(v) + 1e-6
            #         f = v * 0.4 * (1 / (dist - (obs[2] + 100)))
            #         print("f", f)
            #         u += f

            # If near the border, slow down
            tresh = 70
            if (
                abs(pos[0] - pos_bounds[0]) < tresh
                or abs(pos[0] - pos_bounds[1]) < tresh
                or abs(pos[1] - pos_bounds[2]) < tresh
                or abs(pos[1] - pos_bounds[3]) < tresh
            ):
                print("Near border")
                u /= 20.0

            u /= 1000.0
            vx, vy = global_to_local_vel(float(u[0]), float(u[1]), phi_pred)
            # Clip vx and vy to be within the limits (0.05 < abs(v) < 1)
            if abs(vx) < 0.05:
                vx = 0.0
            elif abs(vx) > 1.0:
                vx = 1.0 if vx > 0 else -1.0
            if abs(vy) < 0.05:
                vy = 0.0
            elif abs(vy) > 1.0:
                vy = 1.0 if vy > 0 else -1.0

            # Face the target
            heading_trg = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            heading_pid.set_target(heading_trg)
            phi_err = angle_diff(heading_pid.target, phi_pred)
            w = float(heading_pid.step(phi_pred, phi_err)[0])
            if w > 6.0:
                w = 6.0
            elif w < -6.0:
                w = -6.0
            elif abs(w) < 0.3:
                w = 0.0

            bridge.send(PlayerCmd(player_id, vx, -vy, 0))

            to_save.append(
                {
                    "time": time.time(),
                    "position": pos,
                    "velocity": vel,
                    "orientation": phi,
                    "w": w,
                    "u": u,
                    "heading_pid": heading_pid.get_params(),
                    "pos_pid": pos_pid.get_params(),
                    "phi_err": phi_err,
                    # "ball_pos": ball_pos,
                    "ang_vel": ang_vel,
                    "path": path,
                    "obstacles": obstacles,
                }
            )

            to_sleep = (1 / 20) - dt
            if to_sleep > 0:
                time.sleep(to_sleep)

    except (KeyboardInterrupt, SystemExit) as e:
        pass
    finally:
        print("Exiting Python")
        # Find all file saved as traj##.npy
        idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
        last_idx = max(idxs) if len(idxs) > 0 else 0
        np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
        print("Saved trajectory")

        bridge.send(PlayerCmd(player_id, 0, 0))
