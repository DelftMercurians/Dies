#!/usr/bin/env python3
"""
Simple JAX-based MPC controller for robot navigation.
This script implements a basic MPC using JAX for automatic differentiation and JIT compilation.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
from typing import Tuple, Optional


# MPC Parameters
PREDICTION_HORIZON = 10
CONTROL_HORIZON = 8
DT = 0.05  # 50ms timestep
ROBOT_RADIUS = 90.0  # mm
COLLISION_PENALTY_RADIUS = 200.0  # mm
FIELD_BOUNDARY_MARGIN = 100.0  # mm


def euler_step(pos: jnp.ndarray, vel: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Simple Euler integration step."""
    return pos + vel * dt


def distance_cost(pos: jnp.ndarray, target: jnp.ndarray) -> float:
    """Compute distance-to-target cost."""
    diff = pos - target
    return jnp.sum(diff**2)


def collision_cost(pos: jnp.ndarray, obstacles: jnp.ndarray) -> float:
    """Compute collision avoidance cost."""
    if obstacles.shape[0] == 0:
        return 0.0

    # Compute distances to all obstacles
    diffs = pos[None, :] - obstacles  # Shape: (n_obstacles, 2)
    distances_sq = jnp.sum(diffs**2, axis=1)

    # Penalty increases as we get closer
    penalty_radius_sq = COLLISION_PENALTY_RADIUS**2
    penalties = penalty_radius_sq / (distances_sq + 1.0)

    return jnp.sum(penalties)


def boundary_cost(pos: jnp.ndarray, field_bounds: Optional[jnp.ndarray]) -> float:
    """Compute field boundary violation cost."""
    if field_bounds is None:
        return 0.0

    min_x, max_x, min_y, max_y = field_bounds

    # ReLU-like penalties for boundary violations
    x_lower_violation = jnp.maximum(min_x - pos[0], 0.0)
    x_upper_violation = jnp.maximum(pos[0] - max_x, 0.0)
    y_lower_violation = jnp.maximum(min_y - pos[1], 0.0)
    y_upper_violation = jnp.maximum(pos[1] - max_y, 0.0)

    return (
        x_lower_violation**2
        + x_upper_violation**2
        + y_lower_violation**2
        + y_upper_violation**2
    ) * 100.0


def velocity_constraint_cost(vel: jnp.ndarray, max_vel: float) -> float:
    """Compute velocity constraint violation cost."""
    speed_sq = jnp.sum(vel**2)
    max_speed_sq = max_vel**2
    return jnp.maximum(speed_sq - max_speed_sq, 0.0) * 1000.0


def control_effort_cost(vel: jnp.ndarray) -> float:
    """Compute control effort penalty."""
    return jnp.sum(vel**2) * 0.01


def mpc_cost_function(
    control_sequence: jnp.ndarray,
    initial_pos: jnp.ndarray,
    initial_vel: jnp.ndarray,
    target_pos: jnp.ndarray,
    obstacles: jnp.ndarray,
    field_bounds: Optional[jnp.ndarray],
    max_vel: float,
) -> float:
    """
    Complete MPC cost function.

    Args:
        control_sequence: Control inputs [vx0, vy0, vx1, vy1, ...] shape (2*control_horizon,)
        initial_pos: Starting position [x, y]
        initial_vel: Starting velocity [vx, vy]
        target_pos: Target position [x, y]
        obstacles: Obstacle positions shape (n_obstacles, 2)
        field_bounds: Field boundaries [min_x, max_x, min_y, max_y] or None
        max_vel: Maximum velocity constraint
    """
    total_cost = 0.0
    pos = initial_pos

    # Simulate forward over prediction horizon
    for k in range(PREDICTION_HORIZON):
        # Get control input for this timestep
        control_idx = min(k, CONTROL_HORIZON - 1) * 2
        if k < CONTROL_HORIZON:
            vel = control_sequence[control_idx : control_idx + 2]
        else:
            vel = jnp.array([0.0, 0.0])

        # Euler integration
        pos = euler_step(pos, vel, DT)

        # Add costs
        total_cost += distance_cost(pos, target_pos) * 10.0
        total_cost += collision_cost(pos, obstacles) * 1000.0
        total_cost += boundary_cost(pos, field_bounds)

        # Control effort penalty (only for actual control inputs)
        if k < CONTROL_HORIZON:
            total_cost += control_effort_cost(vel)
            total_cost += velocity_constraint_cost(vel, max_vel)

    return total_cost


# JIT compile the cost function and its gradient for speed
mpc_cost_jit = jit(mpc_cost_function)
mpc_grad_jit = jit(grad(mpc_cost_function, argnums=0))


def solve_mpc(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    field_bounds: Optional[np.ndarray],
    max_vel: float,
    max_iterations: int = 50,
    learning_rate: float = 0.1,
) -> np.ndarray:
    """
    Solve MPC optimization using gradient descent.

    Returns:
        Optimal control input [vx, vy] for the first timestep
    """
    # Convert to JAX arrays
    initial_pos = jnp.array(initial_pos)
    initial_vel = jnp.array(initial_vel)
    target_pos = jnp.array(target_pos)
    obstacles = jnp.array(obstacles) if obstacles.size > 0 else jnp.empty((0, 2))
    field_bounds = jnp.array(field_bounds) if field_bounds is not None else None

    # Initialize control sequence
    control_sequence = jnp.tile(initial_vel, CONTROL_HORIZON)
    control_sequence = jnp.clip(control_sequence, -max_vel, max_vel)

    # Gradient descent optimization
    for _ in range(max_iterations):
        # Compute gradient
        grad_val = mpc_grad_jit(
            control_sequence,
            initial_pos,
            initial_vel,
            target_pos,
            obstacles,
            field_bounds,
            max_vel,
        )

        # Update control sequence
        control_sequence = control_sequence - learning_rate * grad_val

        # Enforce velocity constraints
        control_sequence = jnp.clip(control_sequence, -max_vel, max_vel)

    # Return first control input
    return np.array(control_sequence[:2])


def test_simple_case():
    """Test the MPC with a simple case."""
    print("Testing JAX MPC with simple case...")

    # Simple test case
    initial_pos = np.array([0.0, 0.0])
    initial_vel = np.array([0.0, 0.0])
    target_pos = np.array([1000.0, 500.0])  # 1m forward, 0.5m right
    obstacles = np.array([[500.0, 250.0]])  # One obstacle in the way
    field_bounds = np.array([-2000.0, 2000.0, -1000.0, 1000.0])  # 4m x 2m field
    max_vel = 2000.0  # 2 m/s

    # Solve MPC
    optimal_control = solve_mpc(
        initial_pos, initial_vel, target_pos, obstacles, field_bounds, max_vel
    )

    print(f"Initial position: {initial_pos}")
    print(f"Target position: {target_pos}")
    print(f"Obstacles: {obstacles}")
    print(f"Optimal control: {optimal_control}")
    print(f"Control magnitude: {np.linalg.norm(optimal_control):.2f}")

    return optimal_control


if __name__ == "__main__":
    # Test the implementation
    result = test_simple_case()
    print("JAX MPC test completed successfully!")
