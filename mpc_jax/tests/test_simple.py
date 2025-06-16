#!/usr/bin/env python3
"""Simple tests for MPC JAX implementation"""

import time
import numpy as np
from mpc_jax.main import solve_mpc
from mpc_jax.common import CONTROL_HORIZON


def test_simple_case():
    """Test simple MPC case with one obstacle"""
    n_robots = 2
    initial_pos = np.array([[0.0, 0.0], [10.0, 400.0]])
    initial_vel = np.array([[0.0, 0.0], [0.0, 0.0]])
    target_pos = np.array([[1000.0, 500.0], [-500, 500]])
    obstacles = np.array([[500.0, 250.0]])  # One obstacle in the way
    field_bounds = np.array([-2000.0, 2000.0, -1000.0, 1000.0])
    max_speed = np.array([4000.0, 4000.0])

    optimal_control, _, _, cost = solve_mpc(
        initial_pos,
        initial_vel,
        target_pos,
        obstacles,
        field_bounds,
        max_speed,
        None,
        with_aux=True,
    )

    assert optimal_control is not None
    assert cost < float("inf") and not np.isnan(cost)
    assert optimal_control.shape == (
        n_robots,
        CONTROL_HORIZON,
        2,
    ), f"Expected {(n_robots, CONTROL_HORIZON, 2)} got {optimal_control.shape}"


def test_performance():
    """Test performance with repeated calls"""
    ctrl = test_simple_case()

    times = []
    for i in range(20):
        t = time.time()
        ctrl = test_simple_case()
        elapsed = (time.time() - t) * 1000
        times.append(elapsed)

    # Assert reasonable performance (less than 1 second)
    avg_time = np.mean(np.array(times[10:]))  # Skip warmup
    assert (
        avg_time < 25
    ), f"Average time {avg_time:.1f}ms is too slow, the threshold is 25ms"


if __name__ == "__main__":
    test_simple_case()
    test_performance()
