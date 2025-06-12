import jax
import jax.numpy as jnp
import functools as ft
from common import ROBOT_RADIUS


def distance_cost(pos: jnp.ndarray, target: jnp.ndarray, time_from_now: float):
    dist = jnp.sqrt(jnp.sum((pos - target) ** 2))
    return jnp.clip(
        dist * 2e-2 * (time_from_now + 0.1),
        0,
        100,
    )


def collision_cost(pos: jnp.ndarray, obstacles: jnp.ndarray):
    def single_collision_cost(obstacle: jnp.ndarray, pos: jnp.ndarray):
        nonlocal obstacles
        del obstacles
        assert obstacle.shape == (2,)

        diff = pos[None, :] - obstacle
        distance = jnp.sqrt(jnp.sum(diff**2))

        # Define safety thresholds
        min_safe_distance = 2.1 * ROBOT_RADIUS
        no_cost_distance = 3.5 * ROBOT_RADIUS

        # try to avoid certain collision hard
        danger_zone = distance <= min_safe_distance
        normalized_distance = jnp.clip(distance / min_safe_distance, 0, 1)
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

    return jax.vmap(ft.partial(single_collision_cost, pos=pos))(obstacles).sum()


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


def velocity_constraint_cost(vel: jnp.ndarray, max_vel: float):
    speed_sq = jnp.sum(vel**2)
    max_speed_sq = max_vel**2
    return jnp.maximum(speed_sq - max_speed_sq, 0.0)


def control_effort_cost(vel: jnp.ndarray):
    return jnp.sum(vel**2) * 1e-6
