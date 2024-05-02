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

# Make sure to double check the calculations in this function
def global_to_local_vel(velx, vely, theta) -> tuple:
    '''Convert global velocity to local velocity'''
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


def angle_diff(target, phi) -> float:
    '''Calculate the difference between two angles in radians'''
    return (target - phi + np.pi) % (2 * np.pi) - np.pi

def angle_to_point_to(pos1, pos2) -> float:
    '''Calculate the angle to point to a position from another position in radians'''
    return np.atan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

def normalize_angle(phi, ang_vel) -> float:
    '''Normalize the angle of the robot between -pi and pi'''
    phi_pred = phi + 0.25 * ang_vel
    if phi_pred > pi:
        phi_pred -= 2 * pi
    elif phi_pred < -pi:
        phi_pred += 2 * pi
    return phi_pred

def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4) -> np.ndarray:
    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check if lines are parallel
    if denominator == 0:
        return None  # Lines do not intersect or are coincident
    
    # Calculate t and u
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    
    # Calculate the intersection point using t
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return np.array((px, py))


class VelFilter:
    '''Velocity filter using Savitzky-Golay filter'''
    def __init__(self, dim, window=7, alpha=0.6) -> None:
        self.window = window
        self.state = None
        self.dx = np.zeros(dim)
        self.alpha = alpha

    def update(self, x, dt=(1 / 20)):
        '''Update the velocity filter with new data'''
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
    '''Angular velocity filter using exponential smoothing'''
    def __init__(self) -> None:
        self.prev = None
        self.dx = 0.0

    def update(self, x, dt=(1 / 20)):
        '''Update the angular velocity filter with new data'''
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
    '''PID controller for a system with dim dimensions'''
    def __init__(self, dim, Kp, Kd=0.0, Ki=0.0, diff_func=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dim = dim
        self.integral = np.zeros(dim)
        self.prev_error = np.zeros(dim)
        self.target = None
        self.diff_func = diff_func if diff_func is not None else lambda a, b: a - b

    def set_target(self, target):
        self.target = np.array(target)

    def step(self, current):
        assert self.target is not None, "No target!"
        error = self.diff_func(np.array(self.target), np.array(current))
        self.integral += error
        derivative = (error - self.prev_error)
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

    def get_params(self):
        return np.array([self.Kp, self.Ki, self.Kd])


if __name__ == "__mainPassing__":
    bridge = Bridge()

    player_id = 14
    # Need to change this to the correct teammate id
    # teammate_id = ""

    ball_v_filter = VelFilter(dim=2, alpha=0.9)
    v_filter = VelFilter(dim=2, window=30)
    w_filter = AngVelFilter()

    heading_pid = PID(dim=1, Kp=1.5, Ki=0.0, Kd=0.0, diff_func=angle_diff)
    pos_pid = PID(dim=2, Kp=1.8, Ki=0.0, Kd=0.0)

    dribble = 0
    facing_target = False

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
                print("No own players found")
                continue
            player = next((p for p in msg.own_players if p.id == player_id), None)

            if player is None:
                print("Player not found")
                continue
            if msg.ball is None:
                print("Ball not found")
                continue

            pos_player = np.array([player.position[0], player.position[1]])

            other_players = [p for p in [*msg.own_players, *msg.opp_players] if p.id != player_id]
            pos_teammate = np.array([[p.position[0], p.position[1]] for p in other_players])
            obstacles = [[p.position[0], p.position[1], ROBOT_RADIUS] for p in other_players]

            dt = max(time.time() - last_time, 1e-6)
            last_time = time.time()

            ball_pos = np.array([msg.ball.position[0], msg.ball.position[1]])
            ball_v = ball_v_filter.update(ball_pos)

            # Calculate the slope of the line connecting the two robots
            dx = ball_v[0]
            dy = ball_v[1]
            perpendicular_direction = np.array([-dy, dx])  # A vector perpendicular to the line connecting the robots
            normalized_perpendicular = perpendicular_direction / np.linalg.norm(perpendicular_direction)

            x1, y1 = ball_pos
            x2, y2 = ball_pos - ball_v  # Previous position of the ball
            x3, y3 = pos_player
            x4, y4 = pos_player + normalized_perpendicular

            target = find_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
            
            vel = v_filter.update(pos_player)

            phi = player.orientation
            phi_teammate = other_players.orientation
            ang_vel = w_filter.update(phi)
            phi_pred = normalize_angle(phi, ang_vel)
            

            target[0] = np.clip(target[0], -700, 700) # Clip the target to be within the field (CHANGE THIS)
            target[1] = np.clip(target[1], -500, 500) # Clip the target to be within the field
            pos_pid.set_target(target)
            u = pos_pid.step(pos_player)

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
            heading_trg = np.arctan2(ball_pos[1] - pos_player[1], ball_pos[0] - pos_player[0])
            heading_pid.set_target(-pi/2)
            phi_err = angle_diff(heading_pid.target, phi)
            w = float(heading_pid.step(phi, phi_err)[0])
            if w > 6.0:
                w = 6.0
            elif w < -6.0:
                w = -6.0
            elif abs(w) < 0.3:
                w = 0.0

            if abs(phi_err) > 0.05 and not facing_target:
                facing_target = True
                heading_pid.Ki = 0.05
                bridge.send(PlayerCmd(player_id, 0, 0, w, dribble))
            else:
                heading_pid.Ki = 0.0
                bridge.send(PlayerCmd(player_id, vx, -vy, w, dribble))

            to_save.append(
                {
                    "time": time.time(),
                    "player_time": player.timestamp,
                    "position": pos_player,
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

        bridge.send(PlayerCmd(player_id, 0, 0, 0, 0))