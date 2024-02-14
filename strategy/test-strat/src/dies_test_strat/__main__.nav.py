from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term
import matplotlib.pyplot as plt

from dies_test_strat.RRTStar import RRTStar


print("Starting test-strat")

ROBOT_RADIUS = 90  # mm

pos_bounds = [
    -1400,
    1400,
    -1200,
    1200,
]  # Position Constraints [mm]:    [x_min, x_max, y_min, y_max]


def plan_path(start, goal, obstacles, width, height, margin=3.5):
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        rand_area=[-2000, 15000],
        obstacle_list=[(x, y, r * margin) for x, y, r in obstacles],
        expand_dis=1000,
        robot_radius=100,
        max_iter=2000,
    )
    path = rrt_star.planning(animation=False)

    if path is None:
        raise ValueError("Cannot find path")
    return path


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


def angle_diff(target, phi):
    return (target - phi + np.pi) % (2 * np.pi) - np.pi


class VelFilter:
    def __init__(self, dim, window=7, alpha=0.9) -> None:
        self.window = window
        self.state = None
        self.dx = np.zeros(dim)
        self.alpha = alpha

    def update(self, x, dt=(1 / 20)):
        from scipy.signal import savgol_filter

        if self.state is None:
            self.state = np.tile(x, (self.window, 1))
        else:
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
        if self.prev is None:
            self.prev = x
            return 0.0

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

    player_id = 5

    dest = np.array([800, 800])
    path = None
    path_idx = 0

    ball_v_filter = VelFilter(dim=2)
    v_filter = VelFilter(dim=2, window=30)
    w_filter = AngVelFilter()

    heading_pid = PID(dim=1, Kp=1.7, Ki=0.0002, Kd=0.3, diff_func=angle_diff)
    # heading_pid.set_target(pi)

    pos_pid = PID(dim=2, Kp=0.4, Ki=0.0, Kd=0.0)
    traj_pid = PID(dim=2, Kp=0.8, Ki=0.0, Kd=0.0)
    ball_pid = PID(dim=2, Kp=3, Ki=0.001, Kd=0.0)
    # pos_pid.set_target([0, 0])

    dribble = 300
    dribble_start = None

    ball = None
    to_save = []
    last_time = time.time()
    # bridge.send(PlayerCmd(player_id, 0, 0, 0, dribble))
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
            if abs(player.timestamp - time.time()) > 0.1:
                print("Player timestamp too old")
                continue

            if msg.ball is None:
                print("Ball not found")
                continue
            ball = msg.ball

            dt = max(time.time() - last_time, 1e-6)
            last_time = time.time()

            ball_pos = np.array([ball.position[0], ball.position[1]])
            ball_v = ball_v_filter.update(ball_pos)
            ball_pred = ball_pos + ball_v * 0.07

            pos = np.array([player.position[0], player.position[1]])
            vel = v_filter.update(pos)
            pos_pred = pos + vel * (1 / 20)

            phi = player.orientation
            ang_vel = w_filter.update(phi)
            # predict what the angle will be in 0.2 seconds
            phi_pred = phi + 0.25 * ang_vel
            if phi_pred > pi:
                phi_pred -= 2 * pi
            elif phi_pred < -pi:
                phi_pred += 2 * pi

            other_players = [
                p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id
            ]
            obstacles = [
                [p.position[0], p.position[1], ROBOT_RADIUS] for p in other_players
            ]

            if path is None:
                dest = ball_pos + np.array([0, -150])
                print(f"Creating path, detected obstacles: {len(obstacles)}")
                time.sleep(3)
                path = np.array(
                    plan_path(
                        start=pos,
                        goal=dest,
                        obstacles=obstacles,
                        width=3000,
                        height=3000,
                        margin=3,
                    )
                )
                path = path[::-1]
                print(f"Start {pos}, Destination {dest}")
                print(f"Path created: {path}")

                plt.plot(path[:, 0], path[:, 1], "-o")
                for obs in obstacles:
                    circle = plt.Circle((obs[0], obs[1]), obs[2], color="r", fill=False)
                    plt.gca().add_artist(circle)
                plt.scatter(pos[0], pos[1], c="g", marker="o")
                plt.scatter(dest[0], dest[1], c="r", marker="o")
                plt.show()

                continue

            target = path[path_idx]
            dist = np.linalg.norm(target - pos)
            tresh = 100 if path_idx == len(path) - 1 else 300
            if dist < tresh:
                print(f"Reached target {path_idx}")
                path_idx += 1
                if path_idx >= len(path):
                    print("Reached destination")
                    break
                target = path[path_idx]

            pos_pid.set_target(target)
            u = pos_pid.step(pos)

            line_vec = target - path[path_idx - 1]
            point_vec = pos - path[path_idx - 1]
            scalar_proj = np.dot(point_vec, line_vec) / (
                np.dot(line_vec, line_vec) + 1e-6
            )
            proj = path[path_idx - 1] + scalar_proj * line_vec
            traj_pid.set_target(proj)
            u += traj_pid.step(pos)

            if ball:
                target_ball_pos = ball_pred + np.array([-50, 0])
                ball_pid.set_target(target_ball_pos)
                u += ball_pid.step(ball_pos)

            u /= 1000.0
            vx, vy = global_to_local_vel(float(u[0]), float(u[1]), phi_pred)
            # Clip vx and vy to be within the limits (0.05 < abs(v) < 1)
            max_v = 0.8
            if abs(vx) < 0.05:
                vx = 0.0
            elif abs(vx) > max_v:
                vx = max_v if vx > 0 else -max_v
            if abs(vy) < 0.05:
                vy = 0.0
            elif abs(vy) > max_v:
                vy = max_v if vy > 0 else -max_v

            # Face the target
            # heading_trg = np.arctan2(target[1] - pos[1], target[0] - pos[0])
            heading_pid.set_target(-pi / 2)
            phi_err = angle_diff(heading_pid.target, phi)
            w = float(heading_pid.step(phi, phi_err)[0])
            if w > 6.0:
                w = 6.0
            elif w < -6.0:
                w = -6.0
            elif abs(w) < 0.3:
                w = 0.0

            if dribble_start is None or time.time() - dribble_start > 1.0:
                dribble_start = time.time()
                bridge.send(PlayerCmd(player_id, 0, 0, 0, dribble))
            elif abs(phi_err) > 0.1:
                heading_pid.Ki = 0.05
                bridge.send(PlayerCmd(player_id, 0, 0, w, dribble))
            else:
                heading_pid.Ki = 0.0
                bridge.send(PlayerCmd(player_id, vx, -vy, w, dribble))

            to_save.append(
                {
                    "time": time.time(),
                    "player_time": player.timestamp,
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

    except KeyboardInterrupt as e:
        pass
    finally:
        print("Exiting Python")
        # Find all file saved as traj##.npy
        idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
        last_idx = max(idxs) if len(idxs) > 0 else 0
        np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
        print("Saved trajectory")

        bridge.send(PlayerCmd(player_id, 0, 0, 0, 0))
