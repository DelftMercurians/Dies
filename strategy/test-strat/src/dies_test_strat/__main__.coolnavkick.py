from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term, World
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


def plan_path(start, goal, obstacles, width, height, margin=3):
    rrt_star = RRTStar(
        start=start,
        goal=goal,
        rand_area=[-2000, 15000],
        obstacle_list=[(x, y, r * margin) for x, y, r in obstacles],
        expand_dis=500,
        robot_radius=90,
        max_iter=3000,
        search_until_max_iter=True,
        play_area=[-int(width / 2), int(width / 2), -int(height / 2), int(height / 2)],
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


def find_best_pos(ball_pos, obstacles, offset):
    best_dist = 0
    best_pos = None
    for angle in np.linspace(0, 2 * np.pi, 20):
        pos = ball_pos + np.array([cos(angle), sin(angle)]) * offset
        dist = min([np.linalg.norm(np.array([p[0], p[1]]) - pos) for p in obstacles])
        if dist > best_dist:
            best_dist = dist
            best_pos = pos
    return best_pos


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


dest = np.array([-800, -800])
target_orientation = -pi / 2


class Player:
    def __init__(self, id, bridge):
        self.id = id
        self.bridge = bridge
        self.player = None
        self.u = np.zeros(2)
        self.w = 0.0
        self.dribble = 0

        self.heading_pid = PID(dim=1, Kp=2, Ki=0.02, Kd=0.0, diff_func=angle_diff)
        self.pos_pid = PID(dim=2, Kp=1.2, Ki=0.0, Kd=0.0)
        self.traj_pid = PID(dim=2, Kp=1.2, Ki=0.0, Kd=0.0)
        self.ball_pid = PID(dim=2, Kp=3, Ki=0.001, Kd=0.0)

        self.state = "move_to_ball"

        self.path = None
        self.path_idx = 0

        self.bridge.send(PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, arm=True))

    def update(self, msg: World):
        player = next((p for p in msg.own_players if p.id == self.id), None)
        if player is None:
            print("Player not found")
            return
        self.player = player
        ball = msg.ball
        if ball is None:
            print("No ball")
            return
        self.ball = ball

        print(self.state)
        if self.state == "move_to_ball":
            obstacles = [
                [p.position[0], p.position[1], ROBOT_RADIUS]
                for p in [*msg.own_players, *msg.opp_players]
                if p.id != self.id
            ]
            self.move_to_ball(obstacles)
        elif self.state == "face_ball":
            self.face_ball()
        elif self.state == "get_ball":
            self.get_ball()
        elif self.state == "face_goal":
            self.face_goal()
        elif self.state == "got_ball":
            self.u *= 0
            self.w = 0
            return

    def move_to_ball(self, obstacles):
        pos = np.array(self.player.position)
        phi = self.player.orientation
        ball_pos = np.array([self.ball.position[0], self.ball.position[1]])

        if self.path is None:
            self.dest = find_best_pos(ball_pos, obstacles, 200)
            # obstacles += [[ball_pos[0], ball_pos[1], 30]]
            print(f"Creating path, detected obstacles: {len(obstacles)}")
            self.path = np.array(
                plan_path(
                    start=pos,
                    goal=self.dest,
                    obstacles=obstacles,
                    width=2500,
                    height=2500,
                    margin=1.3,
                )
            )
            self.path = self.path[::-1]
            self.path_idx = 1
            print(f"Start {pos}, Destination {dest}")
            print(f"Path created: {self.path}")
            # plt.plot(self.path[:, 0], self.path[:, 1], "-o")
            # for obs in obstacles:
            #     circle = plt.Circle(
            #         (obs[0], obs[1]), 1.3 * obs[2], color="r", fill=False
            #     )
            #     plt.gca().add_artist(circle)
            # plt.scatter(pos[0], pos[1], c="g", marker="o")
            # plt.scatter(dest[0], dest[1], c="r", marker="o")
            # plt.scatter(ball_pos[0], ball_pos[1], c="b", marker="o")
            # plt.xlim(-2000, 2000)
            # plt.ylim(-2000, 2000)
            # plt.show()

        target = self.path[self.path_idx]

        # Trajectory control
        line_vec = target - self.path[self.path_idx - 1]
        point_vec = pos - self.path[self.path_idx - 1]
        scalar_proj = np.dot(point_vec, line_vec) / (np.dot(line_vec, line_vec) + 1e-6)
        proj = self.path[self.path_idx - 1] + scalar_proj * line_vec
        self.traj_pid.set_target(proj)
        self.u = self.traj_pid.step(pos)

        dist = np.linalg.norm(target - pos)
        if self.path_idx == len(self.path) - 1:
            thresh = 50
            self.pos_pid.set_target(target)
            self.u += self.pos_pid.step(pos)
        else:
            thresh = 300
            # Move forward along the path
            self.u += 600 * line_vec / (np.linalg.norm(line_vec) + 1e-6)
        if dist < thresh:
            print(f"Reached target {self.path_idx}")
            self.path_idx += 1
            if self.path_idx >= len(self.path):
                self.state = "face_ball"
                return
            target = self.path[self.path_idx]

        # Face ball
        self.heading_pid.set_target(0)
        phi_err = angle_diff(self.heading_pid.target, phi)
        if abs(phi_err) > 0.4:
            self.u *= 0
        self.w = float(self.heading_pid.step(phi)[0])

    def face_ball(self):
        pos = np.array(self.player.position)
        phi = self.player.orientation
        ball_pos = np.array([self.ball.position[0], self.ball.position[1]])
        self.heading_pid.set_target(
            np.arctan2(ball_pos[1] - pos[1], ball_pos[0] - pos[0])
        )
        phi_err = angle_diff(self.heading_pid.target, phi)
        if abs(phi_err) > 0.1:
            print(f"Phi error: {phi_err}")
            # self.heading_pid.Kp = 3
            self.heading_pid.Ki = 0.2
            self.u *= 0
            self.w = float(self.heading_pid.step(phi)[0])
        else:
            print("Facing ball")
            self.state = "get_ball"

    def get_ball(self):
        pos = np.array(self.player.position)
        phi = self.player.orientation
        ball_pos = np.array([self.ball.position[0], self.ball.position[1]])
        self.dribble = 300
        self.heading_pid.Ki = 0.0
        self.heading_pid.Kp = 2
        self.w = float(self.heading_pid.step(phi)[0])

        # ball_dir = ball_pos - pos
        # ball_dir /= np.linalg.norm(ball_dir) + 1e-6
        forward_v = np.array([cos(phi), sin(phi)])
        forward_v /= np.linalg.norm(forward_v) + 1e-6
        self.u = 140 * forward_v

        ball_dist = np.linalg.norm(pos - ball_pos)
        ball_disappeared = (self.player.timestamp - self.ball.timestamp) > 0.5
        print("ball_dist", ball_dist)
        print("ball_disappeared", (self.player.timestamp - self.ball.timestamp))
        if ball_dist < 100 or ball_disappeared:
            print("Got ball!")
            self.state = "face_goal"
            self.u *= 0
            self.w = 0

    def face_goal(self):
        pos = np.array(self.player.position)
        phi = self.player.orientation
        self.heading_pid.Kp = 2
        self.heading_pid.Ki = 0
        self.heading_pid.set_target(pi / 2)
        self.pos_pid.set_target(self.dest)
        phi_err = angle_diff(self.heading_pid.target, phi)
        dist = np.linalg.norm(self.dest - pos)
        if abs(phi_err) > 0.06:
            print(f"Phi error: {phi_err}")
            self.w = float(self.heading_pid.step(phi)[0])
        if dist > 50:
            self.u = self.pos_pid.step(pos)

        if abs(phi_err) < 0.06 and dist < 50:
            print("Facing goal")
            self.state = "got_ball"
            time.sleep(2)
            self.kick()

    def step(self):
        if self.player is None:
            return

        u = self.u / 1000.0
        vx, vy = global_to_local_vel(float(u[0]), float(u[1]), self.player.orientation)
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

        w = self.w
        if w > 6.0:
            w = 6.0
        elif w < -6.0:
            w = -6.0
        elif abs(w) < 0.3:
            w = 0.0

        self.bridge.send(PlayerCmd(self.id, vx, -vy, w, self.dribble))

    def kick(self):
        print("Kicking!")
        self.bridge.send(
            PlayerCmd(
                self.id,
                0,
                0,
                0,
                self.dribble,
                kick=True,
            )
        )
        time.sleep(0.05)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, disarm=True)
        )
        time.sleep(0.05)
        self.bridge.send(PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, arm=True))
        time.sleep(0.4)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )
        time.sleep(0.1)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )
        time.sleep(0.1)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )
        time.sleep(0.1)
        self.bridge.send(PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, arm=True))
        time.sleep(0.1)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )
        time.sleep(0.1)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )
        time.sleep(0.1)
        self.bridge.send(
            PlayerCmd(self.id, 0, 0, dribble_speed=self.dribble, kick=True)
        )


if __name__ == "__main__":
    bridge = Bridge()
    player = Player(id=5, bridge=bridge)

    last_time = time.time()
    try:
        while True:
            msg = bridge.recv()
            if not msg:
                continue
            if isinstance(msg, Term):
                break

            player.update(msg)
            player.step()

            dt = max(time.time() - last_time, 1e-6)
            last_time = time.time()
            to_sleep = (1 / 20) - dt
            if to_sleep > 0:
                time.sleep(to_sleep)

    except KeyboardInterrupt as e:
        pass
    finally:
        print("Exiting Python")
        # idxs = [int(fn[4:-4]) for fn in glob("traj[0-9][0-9].npy")]
        # last_idx = max(idxs) if len(idxs) > 0 else 0
        # np.save(f"traj{last_idx + 1:02d}.npy", np.array(to_save))
        # print("Saved trajectory")
