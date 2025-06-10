#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import numpy as np
import optax
import os
import equinox as eqx
from tqdm import tqdm
import functools as ft
from common import (
    ROBOT_RADIUS,
    LEARNING_RATE,
    PREDICTION_HORIZON,
    MAX_ITERATIONS,
    UPDATE_CLIP,
    trajectory_from_control,
)


def euler_step(pos: jnp.ndarray, vel: jnp.ndarray, dt: float) -> jnp.ndarray:
    return pos + vel * dt


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
        min_safe_distance = 2.5 * ROBOT_RADIUS
        no_cost_distance = 4.0 * ROBOT_RADIUS

        # try to avoid certain collision hard
        danger_zone = distance <= min_safe_distance
        normalized_distance = jnp.clip(distance / min_safe_distance, 0, 1)
        danger_factor = jnp.where(danger_zone, 1.1 - normalized_distance, 0.0) * 20

        # try to avoid even getting close to the opponent
        in_decay_zone = jnp.logical_and(
            distance > min_safe_distance, distance <= no_cost_distance
        )
        normalized_distance = jnp.clip(
            (distance - min_safe_distance) / (no_cost_distance - min_safe_distance),
            0,
            1,
        )
        smooth_factor = jnp.where(in_decay_zone, 1 - normalized_distance, 0.0)

        penalties = smooth_factor + danger_factor

        return jnp.sum(penalties)

    return jax.vmap(ft.partial(single_collision_cost, pos=pos))(obstacles).sum()


def boundary_cost(pos: jnp.ndarray, field_bounds: jnp.ndarray | None):
    if field_bounds is None:
        return 0.0

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


def velocity_constraint_cost(vel: jnp.ndarray, max_vel: float):
    speed_sq = jnp.sum(vel**2)
    max_speed_sq = max_vel**2
    return jnp.maximum(speed_sq - max_speed_sq, 0.0)


def control_effort_cost(vel: jnp.ndarray):
    return jnp.sum(vel**2) * 1e-6


def mpc_cost_function(
    control_sequence: jnp.ndarray,
    initial_pos: jnp.ndarray,
    initial_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    obstacles: jnp.ndarray,
    field_bounds: jnp.ndarray | None,
    max_vel: float,
):
    # Generate trajectory from control sequence
    trajectory = trajectory_from_control(control_sequence, initial_pos)

    # Compute position-based costs over the trajectory
    def position_costs_fn(trajectory_point):
        time_from_now, pos_x, pos_y, vel_x, vel_y = trajectory_point
        pos = jnp.array([pos_x, pos_y])
        d_cost = distance_cost(pos, target_pos, time_from_now)
        c_cost = collision_cost(pos, obstacles)
        b_cost = boundary_cost(pos, field_bounds)
        vc_cost = velocity_constraint_cost(jnp.array([vel_x, vel_y]), max_vel)
        return d_cost + c_cost + b_cost + vc_cost

    # Skip initial position (i=0) and compute costs for trajectory steps
    total_position_cost = jax.vmap(position_costs_fn)(trajectory[1:]).sum()

    control_effort_costs = jax.vmap(control_effort_cost)(control_sequence)
    total_control_cost = control_effort_costs.sum()

    return total_position_cost + total_control_cost


mpc_cost_jit = jit(mpc_cost_function)
mpc_grad_jit = jit(jax.value_and_grad(mpc_cost_function, argnums=0))


def clip_vel(vel: jnp.ndarray, max_vel: float):
    speed = jnp.linalg.norm(vel)
    return jnp.where(speed > max_vel, vel * (max_vel / speed), vel)


def solve_mpc_jax(
    initial_pos: jnp.ndarray,
    initial_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    obstacles: jnp.ndarray,
    field_bounds: jnp.ndarray | None,
    max_vel: float,
    max_iterations: int,
    learning_rate: float,
):
    # Initialize control sequence
    control_sequence = jnp.tile(initial_vel, (PREDICTION_HORIZON, 1))
    control_sequence = jax.vmap(lambda vel: clip_vel(vel, max_vel))(control_sequence)

    # Initialize Adam optimizer
    schedule = optax.linear_schedule(
        learning_rate, learning_rate / 10.0, max_iterations
    )
    optimizer = optax.chain(
        optax.sgd(schedule, momentum=0.5, nesterov=True),
        optax.clip_by_global_norm(UPDATE_CLIP),
    )
    opt_state = optimizer.init(control_sequence)

    def optimization_step(carry, _):
        control_seq, opt_state = carry

        # Compute gradient
        cost, grad_val = mpc_grad_jit(
            control_seq,
            initial_pos,
            initial_vel,
            target_pos,
            obstacles,
            field_bounds,
            max_vel,
        )

        # Update with Adam optimizer
        updates, opt_state = optimizer.update(grad_val, opt_state)
        control_seq = optax.apply_updates(control_seq, updates)

        control_seq = jax.vmap(lambda vel: clip_vel(vel, max_vel))(control_seq)

        return (control_seq, opt_state), cost

    (final_control, _), costs = jax.lax.scan(
        optimization_step,
        (control_sequence, opt_state),
        None,
        length=max_iterations,
    )

    # Return first control input
    return final_control


solve_mpc_jitted = eqx.filter_jit(solve_mpc_jax)


def solve_mpc(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    field_bounds: np.ndarray | None,
    max_vel: float,
    max_iterations: int = MAX_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
) -> np.ndarray:
    return np.array(
        solve_mpc_jitted(
            jnp.asarray(initial_pos),
            jnp.asarray(initial_vel),
            jnp.asarray(target_pos),
            jnp.asarray(obstacles),
            jnp.asarray(field_bounds) if field_bounds is not None else None,
            float(max_vel),
            int(max_iterations),
            float(learning_rate),
        ),
    )


def test_simple_case():
    # Simple test case
    initial_pos = np.array([0.0, 0.0])
    initial_vel = np.array([0.0, 0.0])
    target_pos = np.array([1000.0, 500.0])  # 1m forward, 0.5m right
    obstacles = np.array([[500.0, 250.0]])  # One obstacle in the way
    field_bounds = np.array([-2000.0, 2000.0, -1000.0, 1000.0])  # 4m x 2m field
    max_vel = 4000.0  # 2 m/s

    # Solve MPC
    optimal_control = solve_mpc(
        initial_pos, initial_vel, target_pos, obstacles, field_bounds, max_vel
    )

    print(f"Control: {np.linalg.norm(optimal_control):.2f}")
    return optimal_control[0]


if __name__ == "__main__":
    test_simple_case()
    import time

    t = time.time()
    test_simple_case()
    print(f"Took {(time.time() - t) * 1000:.1f}ms")
