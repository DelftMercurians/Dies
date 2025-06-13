import jax
import jax.numpy as jnp
import functools as ft
from .common import ROBOT_RADIUS


def distance_cost(pos: jnp.ndarray, target: jnp.ndarray, time_from_now: float):
    dist = jnp.sum((pos - target) ** 2 + 1e-9)
    return (dist * 1e-2 + jnp.sqrt(dist)) * (time_from_now + 0.5) * 3e-4


def collision_cost(
    pos: jnp.ndarray,
    obstacles: jnp.ndarray,
    mask=None,
    weak_scale=0.0,
    strong_scale=1.0,
):
    if mask is None:
        mask = jnp.ones((len(obstacles),))

    def single_collision_cost(obstacle: jnp.ndarray, pos: jnp.ndarray):
        assert obstacle.shape == (2,), (
            f"Single collision cost: Obstacle must be of shape (2,), but got {obstacle.shape}"
        )
        assert pos.shape == (2,), (
            f"Single collision cost: Pos must be of shape (2,), but got {pos.shape}"
        )

        diff = pos[None, :] - obstacle
        distance = jnp.sqrt(jnp.sum(diff**2) + 1e-9)

        # Define safety thresholds
        min_safe_distance = 2.1 * ROBOT_RADIUS * strong_scale
        no_cost_distance = 3.0 * ROBOT_RADIUS * strong_scale

        no_cost_distance += (no_cost_distance - min_safe_distance) * weak_scale

        # try to avoid certain collision hard
        danger_zone = distance <= min_safe_distance
        normalized_distance = jnp.clip(distance / min_safe_distance, 1e-6, 1)
        danger_factor = jnp.where(danger_zone, 1.1 - normalized_distance, 0.0) * 100

        # try to avoid even getting close to the opponent
        in_decay_zone = jnp.logical_and(
            distance > min_safe_distance, distance <= no_cost_distance
        )
        normalized_distance = jnp.clip(
            (distance - min_safe_distance) / (no_cost_distance - min_safe_distance),
            0,
            1,
        )
        smooth_factor = jnp.where(in_decay_zone, 1 - normalized_distance, 0.0) * 10

        penalties = smooth_factor + danger_factor

        return jnp.sum(penalties)

    obstacle_wise_costs = jax.vmap(ft.partial(single_collision_cost, pos=pos))(
        obstacles
    )
    return (obstacle_wise_costs * mask).sum()


def boundary_cost(pos: jnp.ndarray, field_bounds):
    return 0.0

    """
    min_x, max_x, min_y, max_y = field_bounds

    x_lower_violation = jnp.maximum(min_x - pos[0], 0.0)
    x_upper_violation = jnp.maximum(pos[0] - max_x, 0.0)
    y_lower_violation = jnp.maximum(min_y - pos[1], 0.0)
    y_upper_violation = jnp.maximum(pos[1] - max_y, 0.0)

    return (
        x_lower_violation**2
        + x_upper_violation**2
        + y_lower_violation**2
        + y_upper_violation**2
    )
    """


def velocity_constraint_cost(vel: jnp.ndarray, max_speed: float):
    # we dislike high velocities
    speed_sq = jnp.sum(vel**2)
    max_speed_sq = max_speed**2
    high_cost = jnp.maximum(speed_sq - max_speed_sq, 0.0)

    # and we dislike low velocities
    speed = jnp.sqrt(speed_sq + 1e-9)
    low_cost = jnp.where(
        speed < 100,  # 10cm/s
        100 - speed,
        0,
    )
    return high_cost + low_cost * 0


def control_effort_cost(vel: jnp.ndarray):
    return jnp.sum(vel**2) * 1e-6
