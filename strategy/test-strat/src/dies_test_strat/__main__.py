from math import sqrt, cos, sin, pi
import time
from glob import glob
import numpy as np
from dies_py import Bridge
from dies_py.messages import PlayerCmd, Term
import pygame
import matplotlib.pyplot as plt

from dies_test_strat.vehicle import vehicle_SS
from dies_test_strat.mpc import mpc_control
from dies_test_strat.pathfinder import find_path

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Use qtagg for matplotlib
# import matplotlib

# matplotlib.use("Qt5Agg")

pygame.init()
screen = pygame.display.set_mode((200, 200))

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

class SavGolFilter:
    """Uses a savitky-golay filter to compute velocity from position"""

    def __init__(self, dim, window=7) -> None:
        # self.window = window
        # self.state = np.zeros((self.window, dim))
        self.prev = 0.0
        self.dx = 0.0
        pass

    def update(self, x, dt):
        # from scipy.signal import savgol_filter

        # self.state = np.roll(self.state, -1, axis=0)
        # self.state[-1] = np.array(x)
        # dx = savgol_filter(self.state, self.window, 2, deriv=1, axis=0)
        diff = x - self.prev
        self.prev = x
        if diff > pi:
            diff -= 2*pi
        elif diff < -pi:
            diff += 2*pi
        # print(f"{diff=}")
        # if dt > 0.0:
        self.dx = self.dx * 0.8 + 0.2 * (diff * 20)
        # else:
        #     print("dt = 0.0 !!!!!!!!!!!!!!!!!!!!")

        return self.dx
    

class BasicDerivative:
    """Uses a savitky-golay filter to compute velocity from position"""

    def __init__(self, dim, window=7) -> None:
        # self.window = window
        self.prev = np.zeros((dim))
        self.dx =np.zeros((dim))
        pass

    def update(self, x, dt):
        # from scipy.signal import savgol_filter

        # self.state = np.roll(self.state, -1, axis=0)
        # self.state[-1] = np.array(x)
        # dx = savgol_filter(self.state, self.window, 2, deriv=1, axis=0)
        diff = x - self.prev
        self.prev = x
        # print(f"{diff=}")
        # if dt > 0.0:
        self.dx = self.dx * 0.5 + 0.5 * (diff * 20)
        # else:
        #     print("dt = 0.0 !!!!!!!!!!!!!!!!!!!!")

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
        # self.integral = np.zeros(self.dim)
        self.target = np.array(target)

    def step(self, current, error=None):
        assert self.target is not None, "No target!"
        if error is None:
            error = self.diff_func(np.array(self.target), np.array(current))
        else:
            error = np.array(error)
        self.integral += error
        derivative = self.diff_func(current, self.prev_inp)
        self.prev_inp = current
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def get_params(self):
        return np.array([self.Kp, self.Ki, self.Kd])


def angle_diff(target, phi):
    return (target - phi + np.pi) % (2 * np.pi) - np.pi


def mypause(interval):
    backend = plt.rcParams["backend"]
    if backend in matplotlib.rcsetup.interactive_bk:
        figManager = matplotlib._pylab_helpers.Gcf.get_active()
        if figManager is not None:
            canvas = figManager.canvas
            if canvas.figure.stale:
                canvas.draw()
            canvas.start_event_loop(interval)
            return


if __name__ == "__main__":
    bridge = Bridge()

    w_filter = SavGolFilter(dim=1)
    ball_filter = BasicDerivative(dim=2)

    # player_id = 5
    # rid = 3

    player_id = 14
    rid = 2

    # heading_pid = PID(dim=1, Kp=0.25, Ki=0.015, Kd=2.6)
    # heading_Kp_base = 1.5
    # heading_pid = PID(dim=1, Kp=heading_Kp_base, Ki=0.03, Kd=2.4)
    # heading_pid.set_target(0)

    # pos_pid = PID(dim=2, Kp=0.1, Ki=0.03, Kd=0.1)
    heading_Kp_base = 1.3
    heading_pid = PID(dim=1, Kp=heading_Kp_base, Ki=0.05, Kd=-0.0, diff_func=angle_diff)
    heading_pid.set_target(pi)

    pos_pid = PID(dim=2, Kp=0.4, Ki=0.05, Kd=0.0)
    pos_pid.set_target([0, 0])

    targets = np.array(
        [
            [-1000, 0],
            # [700, 700],
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
    target_dir = np.array([0, 0], dtype=np.float64)
    try:
        while True:
            pygame.event.pump()  # process event queue
            # Check for quit events
            quit = next(
                (True for e in pygame.event.get() if e.type == pygame.QUIT), False
            )
            if quit:
                break

            # print("Running")
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

            if not f_init:
                f.x = np.array([player.position[0], player.position[1], 0, 0])
                f_init = True

            # other_players = [
            #     p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id
            # ]
            # if len(other_players) == 0:
            #     print("No other players found")
            #     continue

            # Get pressed keys
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                heading_pid.set_target(pi/2)
            if keys[pygame.K_RIGHT]:
                heading_pid.set_target(3*pi/2)
            if keys[pygame.K_UP]:
                heading_pid.set_target(0)
            if keys[pygame.K_DOWN]:
                heading_pid.set_target(pi)

            if (
                not keys[pygame.K_LEFT]
                and not keys[pygame.K_RIGHT]
                and not keys[pygame.K_UP]
                and not keys[pygame.K_DOWN]
            ):
                target_dir = np.array([0, 0], dtype=np.float64)
            target_dir /= np.linalg.norm(target_dir) + 1e-6
            u = target_dir * 1300

            ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
            # ball_pos_avg[idx] = ball_pos
            # ball_pos = ball_pos_avg.mean(axis=0)

            idx = (idx + 1) % ball_n
            pos = np.array([player.position[0], player.position[1]])
            vel = np.array([player.velocity[0], player.velocity[1]])
            phi = player.orientation
            dt = (time.time_ns()  - last_time) / 1e9
            ang_vel = w_filter.update(phi, dt)
            ball_vel = ball_filter.update(ball_pos, dt)
            phi_pred = phi + 0.25 * ang_vel # predict what the angle will be in 0.2 seconds
            ball_pred = ball_pos + 0.25 * ball_vel # predict what the angle will be in 0.2 seconds
            if phi_pred > pi:
                phi_pred -= 2*pi
            elif phi_pred < -pi:
                phi_pred += 2*pi

            # target = targets[target_idx]
            # target = ball_pos + np.array([0, +40])
            # dist = np.linalg.norm(pos - target)
            # target_dir = find_path(
            #     pos,
            #     vel,
            #     target,
            #     static_obstacles=[
            #         np.array([p.position[0], p.position[1]]) for p in other_players
            #     ],
            # )
            plan = None
            # target_dir, _, plan = mpc_control(
            #     N=10,
            #     dt=1 / 5,
            #     tw=0,
            #     obstacles=[[p.position[0], p.position[1], 100] for p in other_players],
            #     move_obstacles=[],
            #     last_plan=None,
            #     x_init=pos,
            #     u_init=vel,
            #     x_target=target,
            #     pos_constraints=pos_bounds,
            #     vel_constraints=[-1000, 1000, -1000, 1000],
            # )
            # target_dir /= np.linalg.norm(target_dir)
            # u = target_dir * dist * pos_pid.Kp
            # pos_pid.set_target(target)
            # u_pid = pos_pid.step(pos)
            # if dist < 200:
            #     u = u_pid

            
            # f.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
            # f.Q = Q_discrete_white_noise(dim=4, dt=dt, var=0.013)
            # f.predict()
            # f.update(pos)

            # if dist < 50:
            #     # target_idx = (target_idx + 1) % len(targets)
            #     done_facing = False
            #     # start_time = None
            #     print("Reached target")
            #     continue

            # Face the ball
            heading = np.arctan2(ball_pred[1] - pos[1], ball_pred[0] - pos[0])
            # Face forward
            # heading = np.arctan2(target_dir[1] - pos[1], target_dir[0] - pos[0])
            # heading += pi / 2
            heading_pid.set_target(heading)
            # The error in angle between phi and the target heading (-pi, pi)
            err = angle_diff(heading_pid.target, phi_pred)
            if abs(err) < 0.1:
                start_time = time.time()
                done_facing = True
            # elif abs(err) > 0.5:
            #     done_facing = False
            #     start_time = None

            if done_facing:
                vel_err = u - vel
                u = u + vel_err * 0.2
                u /= 1000.0
                vx, vy = global_to_local_vel(
                    float(u[0]), float(u[1]), phi_pred + (w * dt / 2)
                )
                v = np.linalg.norm(u)
                heading_pid.Kp = 2.5
                heading_pid.Kd = 0.0
                heading_pid.Ki = 0.0
                w = float(heading_pid.step(phi_pred, err)[0])
                if w > 6.0:
                    w = 6.0
                elif w < -6.0:
                    w = -6.0
                elif abs(w) < 0.3:
                    w = 0.0
                bridge.send(PlayerCmd(rid, vx, -vy, w))

                print(f"{phi}\t{ang_vel}\t{phi_pred}")
            else:
                heading_pid.Kd = 0.0
                heading_pid.Ki = 0.0
                w = float(heading_pid.step(phi_pred, err)[0])
                bridge.send(PlayerCmd(rid, 0, 0, w))

            # Live plot w
            # plt.plot(time.time(), w, "ro")
            # mypause(0.05)

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
                    "phi_err": err,
                    # "ball_pos": ball_pos,
                    # "plan": plan,
                }
            )
            last_time = time.time_ns() 

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

        pygame.quit()
        bridge.send(PlayerCmd(rid, 0, 0))
