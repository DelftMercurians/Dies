from math import sqrt, cos, sin, pi
import time
from glob import glob
from dies_test_strat.PRMController import PRMController
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term
import matplotlib.pyplot as plt

# from rrtplanner import perlin_occupancygrid, RRTStar, random_point_og


print("Starting test-strat")

ROBOT_RADIUS = 90  # mm

pos_bounds = [
    -1400,
    1400,
    -1200,
    1200,
]  # Position Constraints [mm]:    [x_min, x_max, y_min, y_max]


def plan_path(start, goal, obstacles, width, height, margin=3.5):
    center = np.array([width, height]) // 2
    n = 2000
    r_rewire = 160
    og = perlin_occupancygrid(width, height, 0)
    for x, y, radius in obstacles:
        radius = int(radius * margin)
        x += center[0]
        y += center[1]
        start_x = int(max(0, x - radius))
        end_x = int(min(og.shape[0], x + radius))
        start_y = int(max(0, y - radius))
        end_y = int(min(og.shape[1], y + radius))
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if (i - x) ** 2 + (j - y) ** 2 <= radius**2:
                    og[i, j] = 1

    start = start + center
    goal = goal + center

    rrts = RRTStar(og, n, r_rewire)
    T, gv = rrts.plan(start, goal)
    path = rrts.route2gv(T, gv)
    path_pts = rrts.vertices_as_ndarray(T, path)

    path_pts -= center
    return path_pts


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


def angle_diff(target, phi):
    return (target - phi + np.pi) % (2 * np.pi) - np.pi


class VelFilter:
    def __init__(self, dim, window=7, alpha=0.6) -> None:
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


class Keeper:

    def __init__(self, player_id, bridge):
        self.player_id = player_id
        self.bridge = bridge
        self.ball_v_filter = VelFilter(dim=2, alpha=0.9)
        self.v_filter = VelFilter(dim=2, window=30)
        self.w_filter = AngVelFilter()

        self.heading_pid = PID(dim=1, Kp=1.5, Ki=0.0, Kd=0.0, diff_func=angle_diff)
        # self.heading_pid.set_target(pi)

        self.pos_pid = PID(dim=2, Kp=1.8, Ki=0.0, Kd=0.0)
        # traj_pid = PID(dim=2, Kp=1.5, Ki=0.0, Kd=0.0)
        # ball_pid = PID(dim=2, Kp=0.2, Ki=0.0, Kd=0.0)
        # self.pos_pid.set_target([0, 0])

        self.dribble = 0

        self.facing_target = False

    def update(self, msg, last_time):
        if len(msg.own_players) == 0:
            print("No own players not found")
            return
        player = next((p for p in msg.own_players if p.id == self.player_id), None)
        if player is None:
            print("Player not found")
            return
        if abs(player.timestamp - time.time()) > 0.1:
            print(f"Player timestamp too old {player.timestamp - time.time()}")
            return
        if msg.ball is None:
            print("Ball not found")
            return
        pos = np.array([player.position[0], player.position[1]])

        other_players = [
            p for p in [*msg.own_players, *msg.opp_players] if p.id != self.player_id
        ]
        obstacles = [
            [p.position[0], p.position[1], ROBOT_RADIUS] for p in other_players
        ]

        # if path is None:
        #     print(f"Creating PRM, detected obstacles: {len(obstacles)}")
        #     path = plan_path(
        #         start=pos, goal=dest, obstacles=obstacles, width=3000, height=3000
        #     )
        #     path = path[:, 1, :]
        #     print(f"Start {pos}, Destination {dest}")
        #     print(f"Path created: {path}")
        #     continue

        dt = max(time.time() - last_time, 1e-6)
        last_time = time.time()

        ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
        ball_v = self.ball_v_filter.update(ball_pos)
        ball_pred = ball_pos + ball_v * 1.0

        vel = self.v_filter.update(pos)
        pos_pred = pos + vel * (1 / 20)

        phi = player.orientation
        ang_vel = self.w_filter.update(phi)
        # predict what the angle will be in 0.2 seconds
        phi_pred = phi + 0.25 * ang_vel
        if phi_pred > pi:
            phi_pred -= 2 * pi
        elif phi_pred < -pi:
            phi_pred += 2 * pi

        # dist = np.linalg.norm(target - pos)
        # if dist < 150:
        #     print(f"Reached target {path_idx}")
        #     path_idx += 1
        #     if path_idx >= len(path):
        #         print("Reached destination")
        #         break
        #     target = path[path_idx]

        if ball_v[1] > 500:
            t_intersect = (1200 - ball_pred[1]) / ball_v[1]

            # Calculate intersection point
            intersection_x = ball_pred[0] + ball_v[0] * t_intersect * 1.2
            # dist_x = intersection_x - pos
            print(f"AAAAAAAAAAAA {t_intersect} {ball_v[1]}")
        elif ball_pred[1] < 500:
            a = (1400 - 1200) / (1400 - ball_pred[1])
            intersection_x = ball_pred[0] * a
        else:
            intersection_x = ball_pred[0]

        target = np.array([intersection_x, 1200])
        target[0] = np.clip(target[0], -700, 700)
        self.pos_pid.set_target(target)
        u = self.pos_pid.step(pos)

        # line_vec = target - path[path_idx - 1]
        # point_vec = pos - path[path_idx - 1]
        # scalar_proj = np.dot(point_vec, line_vec) / np.dot(line_vec, line_vec)
        # proj = path[path_idx - 1] + scalar_proj * line_vec
        # traj_pid.set_target(proj)
        # u += traj_pid.step(pos)

        # if ball:
        #     target_ball_pos = ball_pred + np.array([-100, 0])
        #     ball_pid.set_target(target_ball_pos)
        #     u += ball_pid.step(ball_pos)

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
        heading_trg = np.arctan2(ball_pred[1] - pos[1], ball_pred[0] - pos[0])
        self.heading_pid.set_target(-pi / 2)
        phi_err = angle_diff(self.heading_pid.target, phi)
        w = float(self.heading_pid.step(phi, phi_err)[0])
        if w > 6.0:
            w = 6.0
        elif w < -6.0:
            w = -6.0
        elif abs(w) < 0.3:
            w = 0.0

        if abs(phi_err) > 0.05 and not self.facing_target:
            self.facing_target = True
            self.heading_pid.Ki = 0.05
            self.bridge.send(PlayerCmd(self.player_id, 0, 0, w, self.dribble))
        else:
            self.heading_pid.Ki = 0.0
            self.bridge.send(PlayerCmd(self.player_id, vx, -vy, w, self.dribble))
