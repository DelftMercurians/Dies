from math import sqrt, cos, sin, pi
import time
from dies_py import Bridge
from dies_py.messages import PlayerCmd

print("Starting test-strat")

# PID coefficients
Kp = 0.1  # Proportional gain
Ki = 0.0  # Integral gain
Kd = 0.0  # Derivative gain

# PID variables
integral = [0, 0]
prev_error = [0, 0]


def global_to_local_vel(velx, vely, theta):
    # theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


def traj(t):
    speed = 1
    radius = 250
    t_scaled = speed * t

    x = radius * cos(t_scaled)
    y = radius * sin(t_scaled)

    # x += 500

    return x, y


if __name__ == "__main__":
    bridge = Bridge()
    tolerance = 50
    start_time = time.time()
    # is_going_to_center = True

    while True:
        msg = bridge.recv()
        if not msg:
            continue
        # bridge.send(PlayerCmd(0, 0, 0, 0))

        if len(msg.own_players) == 0:
            print("No players not found")
            continue
        player = msg.own_players[0]
        rid = player.id
        print(player.position)

        bridge.send(PlayerCmd(rid, 0, 0, 0))
        time.sleep(1 / 60)  # Small delay for the control loop
        continue

        current_time = time.time() - start_time
        target_pos = (0, 0)

        error = (
            target_pos[0] - player.position[0],
            target_pos[1] - player.position[1],
        )
        integral[0] += error[0]
        integral[1] += error[1]
        derivative = (error[0] - prev_error[0], error[1] - prev_error[1])

        output_x = Kp * error[0] + Ki * integral[0] + Kd * derivative[0]
        output_y = Kp * error[1] + Ki * integral[1] + Kd * derivative[1]

        prev_error = error

        dist = sqrt(error[0] ** 2 + error[1] ** 2)
        if dist > tolerance:
            x, y = global_to_local_vel(output_x, output_y, player.orientation)
            # bridge.send(PlayerCmd(rid, x, y, 0))
            bridge.send(PlayerCmd(rid, 0, 0, 100))
            time.sleep(1 / 60)  # Small delay for the control loop
        # elif is_going_to_center:
        #     # is_going_to_center = False
        #     bridge.send(PlayerCmd(rid, 0, 0, 0))
        #     time.sleep(2)
        #     start_time = time.time()
