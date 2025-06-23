#!/usr/bin/env python3
"""Simple tests for MPC JAX implementation"""

import time
import numpy as np
from mpc_jax.main import solve_mpc
from mpc_jax.common import Result, CONTROL_HORIZON


def simple_case(last_solution=None):
    n_robots = 2
    initial_pos = np.array([[5.0, 10.0], [80.0, 700.0]])
    initial_vel = np.array([[-50.0, 10.0], [20.0, 30.0]])
    target_pos = np.array([[1000.0, 500.0], [-500, 500]])
    obstacles = np.array([[500.0, -250.0]])  # One obstacle in the way
    field_bounds = np.array([-2000.0, 2000.0, -1000.0, 1000.0])
    max_speed = np.array([4000.0, 4000.0])
    ball_pos = None

    r = solve_mpc(
        initial_pos,
        initial_vel,
        target_pos,
        obstacles,
        ball_pos,
        field_bounds,
        max_speed,
        last_solution,
    )

    assert r.u is not None
    assert not np.isinf(r.cost) and not np.isnan(r.cost), (
        "MPC failed (no collision-free resolution) for a simple case"
    )
    assert r.u.shape == (
        n_robots,
        CONTROL_HORIZON,
        2,
    ), f"Expected {(n_robots, CONTROL_HORIZON, 2)} got {r.u.shape}"
    return r.u, r.cost


def test_simple_case():
    simple_case()


def test_performance():
    last_solution, st_cost = simple_case()

    times = []
    costs = []
    for _ in range(50):
        t = time.time()

        sol, cost = simple_case(last_solution)
        assert np.any(sol != last_solution)
        costs.append(cost)
        last_solution = sol

        elapsed = (time.time() - t) * 1000
        times.append(elapsed)

    avg_time = np.mean(np.array(times[20:]))
    assert avg_time < 25, (
        f"Average time {avg_time:.1f}ms is too slow, the threshold is 25ms"
    )
    print(
        f"\n\n\tAn MPC run is expected to take {avg_time:.1f}ms \n\tAverage score was {np.mean(costs):.1f}, down from {st_cost:.1f} after one iteration.\n"
    )


if __name__ == "__main__":
    test_simple_case()
    test_performance()
