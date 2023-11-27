from math import sqrt, cos, sin, degrees, pi
from dies_py import Bridge
from dies_py.messages import PlayerCmd

print("Starting test-strat")

# PID coefficients
Kp = 1  # Proportional gain
Ki = 0.000001  # Integral gain
Kd = 0.05  # Derivative gain

# PID variables
integral = [0, 0]
prev_error = [0, 0]


def global_to_local_vel(velx, vely, theta):
    theta = theta + pi / 2
    new_x = vely * sin(theta) + velx * cos(theta)
    new_y = -vely * cos(theta) + velx * sin(theta)
    return new_x, new_y


if __name__ == "__main__":
    bridge = Bridge()
    rid = 0
    target_pos = (0, 0)
    tolerance = 10

    while True:
        msg = bridge.recv()
        if msg:
            player = next((p for p in msg.own_players if p.id == rid), None)
            if player is None:
                print("Player not found")
                continue

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
                bridge.send(PlayerCmd(rid, x, y, 0))
            else:
                bridge.send(PlayerCmd(rid, 0, 0, 0))
