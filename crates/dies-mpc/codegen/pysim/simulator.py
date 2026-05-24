"""Open-loop simulator and cost-shape inspector.

`rollout` is the bread-and-butter dynamics replay: given a control sequence
and a heading trajectory, roll the model forward and return states.

`cost_along` evaluates the cost over a given trajectory plus a per-stage
breakdown — useful for understanding which cost term dominates at which
stage when tuning weights.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .types import MpcTarget, RobotParams, Vec2


def rollout(
    step_fn: Callable,
    x0: np.ndarray,
    controls: np.ndarray,
    headings: np.ndarray,
    dt: float,
    params: RobotParams,
) -> np.ndarray:
    """Forward-Euler rollout of `controls` from `x0`.

    Args:
        step_fn: from `build(dynamics_model)["step"]`.
        x0: shape (4,) — initial state [px, py, vx, vy].
        controls: shape (N, 2) — commanded velocities.
        headings: shape (>=N,) — per-stage heading in radians.
        dt: stage duration in seconds.
        params: dynamics parameters.

    Returns:
        states: shape (N+1, 4).
    """

    x0 = np.asarray(x0, dtype=float).reshape(4)
    n = len(controls)
    states = np.empty((n + 1, 4), dtype=float)
    states[0] = x0
    for k in range(n):
        u_k = np.asarray(controls[k], dtype=float)
        h_k = float(headings[k]) if k < len(headings) else 0.0
        states[k + 1] = step_fn(states[k], u_k, h_k, dt, params)
    return states


def cost_along(
    cost_callables: dict[str, Callable],
    states: np.ndarray,
    controls: np.ndarray,
    target: MpcTarget,
    u_prev_first: np.ndarray | None = None,
) -> tuple[float, list[dict]]:
    """Evaluate stage cost over a given trajectory.

    Returns:
        total: scalar total cost (= sum of stage costs).
        per_stage: list of length N with per-stage dicts carrying `stage`
            cost and the per-term breakdown.
    """

    stage_scalar = cost_callables["stage_cost_scalar"]

    n = len(controls)
    if u_prev_first is None:
        u_prev_first = np.zeros(2)
    u_prev = np.asarray(u_prev_first, dtype=float)

    per_stage: list[dict] = []
    total = 0.0
    pos = np.array([target.p.x, target.p.y])
    vel = np.array([target.v.x, target.v.y])
    w = target.weights

    for k in range(n):
        x_k = states[k]
        u_k = np.asarray(controls[k], dtype=float)
        cost = float(stage_scalar(x_k, u_k, u_prev, target))
        d_pos = x_k[:2] - pos
        d_vel = x_k[2:] - vel
        d_u = u_k - u_prev
        per_stage.append({
            "k": k,
            "stage": cost,
            "pos": 0.5 * w.position * float(d_pos @ d_pos),
            "vel": 0.5 * w.velocity * float(d_vel @ d_vel),
            "ctrl": 0.5 * w.control * float(u_k @ u_k),
            "dctrl": 0.5 * w.control_smoothness * float(d_u @ d_u),
        })
        total += cost
        u_prev = u_k

    return total, per_stage
