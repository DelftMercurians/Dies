import jax
import jax.numpy as jnp
import functools as ft
from .common import ROBOT_RADIUS, BALL_RADIUS


def distance_cost(pos: jnp.ndarray, target: jnp.ndarray, time_from_now: float):
    dist = jnp.sum((pos - target) ** 2 + 1e-9)
    return jnp.sqrt(dist) * 5e-4


def collision_cost(
    pos: jnp.ndarray,
    obstacles: jnp.ndarray,
    min_safe_distance,
    no_cost_distance,
    mask=None,
):
    if mask is None:
        mask = jnp.ones((len(obstacles),))

    def single_collision_cost(obstacle: jnp.ndarray, pos: jnp.ndarray):
        assert (
            obstacle.shape == (2,)
        ), f"Single collision cost: Obstacle must be of shape (2,), but got {obstacle.shape}"
        assert pos.shape == (
            2,
        ), f"Single collision cost: Pos must be of shape (2,), but got {pos.shape}"

        diff = pos[None, :] - obstacle
        distance = jnp.sqrt(jnp.sum(diff**2) + 1e-9)

        # try to avoid certain collision hard
        danger_zone = distance <= min_safe_distance
        normalized_distance = jnp.clip(distance / min_safe_distance, 1e-6, 1)
        danger_factor = jnp.where(danger_zone, 2.0 - normalized_distance, 0.0) * 200

        # try to avoid even getting close to the opponent
        in_decay_zone = jnp.logical_and(
            distance > min_safe_distance, distance <= no_cost_distance
        )
        normalized_distance = jnp.clip(
            (distance - min_safe_distance) / (no_cost_distance - min_safe_distance),
            0,
            1,
        )
        smooth_factor = (
            jnp.where(in_decay_zone, (1 - normalized_distance) ** 2, 0.0) * 100.0
        )

        penalties = smooth_factor + danger_factor

        return jnp.sum(penalties)

    obstacle_wise_costs = jax.vmap(ft.partial(single_collision_cost, pos=pos))(
        obstacles
    )
    return (obstacle_wise_costs * mask).sum()


def ball_collision_cost(
    pos: jnp.ndarray,
    ball_pos: jnp.ndarray,
    min_safe_distance,
    no_cost_distance,
):
    assert ball_pos.shape == (
        2,
    ), f"Ball collision cost: Ball pos must be of shape (2,), but got {ball_pos.shape}"
    assert pos.shape == (
        2,
    ), f"Ball collision cost: Robot pos must be of shape (2,), but got {pos.shape}"

    return collision_cost(
        pos,
        ball_pos[None, :],
        min_safe_distance=min_safe_distance,
        no_cost_distance=no_cost_distance,
    )


def boundary_cost(pos: jnp.ndarray, field_bounds, avoid_goal_area):
    """
    Calculate cost for being inside penalty areas or outside field boundaries.
    field_bounds: FieldBounds object with field dimensions
    """
    half_length = field_bounds.field_length / 2.0
    half_width = field_bounds.field_width / 2.0

    x, y = pos[0], pos[1]
    cost = 0.0

    # Cost for being outside field boundaries (very high penalty)
    x_outside = jnp.maximum(jnp.abs(x) - half_length, 0.0)
    y_outside = jnp.maximum(jnp.abs(y) - half_width, 0.0)
    field_violation = x_outside + y_outside
    cost += field_violation * 1000.0  # Very high penalty for leaving field

    def goal_area_cost():
        # Cost for being inside penalty areas (amplifies with depth)
        # Left penalty area
        left_penalty_x_inside = jnp.maximum(
            -half_length + field_bounds.penalty_area_depth - x, 0.0
        )
        left_penalty_y_inside = jnp.maximum(
            field_bounds.penalty_area_width / 2.0 - jnp.abs(y), 0.0
        )
        left_penalty_violation = jnp.minimum(
            left_penalty_x_inside, left_penalty_y_inside
        )
        left_penalty_violation = jnp.maximum(left_penalty_violation, 0.0)

        # Right penalty area
        right_penalty_x_inside = jnp.maximum(
            x - (half_length - field_bounds.penalty_area_depth), 0.0
        )
        right_penalty_y_inside = jnp.maximum(
            field_bounds.penalty_area_width / 2.0 - jnp.abs(y), 0.0
        )
        right_penalty_violation = jnp.minimum(
            right_penalty_x_inside, right_penalty_y_inside
        )
        right_penalty_violation = jnp.maximum(right_penalty_violation, 0.0)

        # Sum of absolute distances as penalty (deeper = higher cost)
        penalty_area_cost = (left_penalty_violation + right_penalty_violation) * 10.0
        return penalty_area_cost

    cost += jax.lax.cond(avoid_goal_area, goal_area_cost, lambda: 0.0)

    return cost


def velocity_constraint_cost(vel: jnp.ndarray, max_speed: float):
    # we dislike high velocities
    speed_sq = jnp.sum(vel**2)
    max_speed_sq = max_speed**2
    high_cost = jnp.maximum(speed_sq - max_speed_sq, 0.0)
    return high_cost
