import numpy as np
from typing import List, Tuple

RADIUS_ROBOT = 0.0793


def get_relative_speed(pos1, pos2, vel1, vel2) -> float:
    d = pos1 - pos2
    d /= np.linalg.norm(d) + 1e-6
    s = vel1 - vel2
    return np.abs(np.dot(s, d))


def find_path(
    start_pos,
    vel,
    goal,
    static_obstacles,
    alpha=0.02,
    beta=20,
    influence_factor: Tuple[int, int] = (5, 1),
):
    obstacles = []
    for pos in static_obstacles:
        obstacles.append(
            (
                pos,
                get_relative_speed(pos, start_pos, 0, vel),
            )
        )

    f = goal - start_pos
    dist = np.linalg.norm(f)
    f /= dist + 1e-6
    attractive_force = alpha * dist
    f *= attractive_force

    base_factor, speed_factor = influence_factor

    influence_radius = (
        base_factor * RADIUS_ROBOT * 1000
    )  # 1000 -> convert the unit to mm

    for o, speed in obstacles:
        d = np.linalg.norm(start_pos - o)
        final_influence_radius = influence_radius + speed * speed_factor
        if d < final_influence_radius:
            repulsive_force = (
                1.0 / (d - 2 * RADIUS_ROBOT * 1000)
                - 1.0 / (final_influence_radius - 2 * RADIUS_ROBOT * 1000)
            ) * (speed * beta + 1)
            fd = start_pos - o
            fd /= np.linalg.norm(fd) + 1e-6
            f += fd * repulsive_force

    # normalize
    f /= np.linalg.norm(f) + 1e-6
    return f
