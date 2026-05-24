"""Python port of `crates/dies-mpc/src/solver.rs` — iLQR with multi-start.

The math mirrors the Rust implementation 1:1. Both consume the same
symbolic dynamics + cost via lambdified callables, so a divergence between
the two is purely an algorithmic-port bug. The parity test in
`tests/test_parity.py` keeps this honest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .types import MpcTarget, RobotParams, SolverConfig, SolveResult, Vec2


_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625)


@dataclass
class _Backward:
    k_fb: np.ndarray         # shape (N, 2)         feedforward
    kk_fb: np.ndarray        # shape (N, 2, 4)      feedback
    expected_dv1: float
    expected_dv2: float


def _rollout(
    step_fn: Callable,
    stage_scalar: Callable,
    x0: np.ndarray,
    controls: np.ndarray,
    u_prev_first: np.ndarray,
    heading_traj: np.ndarray,
    target: MpcTarget,
    params: RobotParams,
    dt: float,
) -> tuple[np.ndarray, float]:
    n = len(controls)
    states = np.empty((n + 1, 4))
    states[0] = x0
    cost = 0.0
    u_prev = u_prev_first
    for k in range(n):
        u_k = controls[k]
        h_k = float(heading_traj[k]) if k < len(heading_traj) else 0.0
        cost += float(stage_scalar(states[k], u_k, u_prev, target))
        states[k + 1] = step_fn(states[k], u_k, h_k, dt, params)
        u_prev = u_k
    return states, cost


def _backward_pass(
    dyn: dict[str, Callable],
    cost: dict[str, Callable],
    states: np.ndarray,
    controls: np.ndarray,
    u_prev_first: np.ndarray,
    heading_traj: np.ndarray,
    target: MpcTarget,
    params: RobotParams,
    dt: float,
    reg: float,
) -> _Backward | None:
    n = len(controls)
    v_x = np.zeros(4)
    v_xx = np.zeros((4, 4))

    k_fb = np.zeros((n, 2))
    kk_fb = np.zeros((n, 2, 4))
    dv1 = 0.0
    dv2 = 0.0

    I2 = np.eye(2)

    for k in reversed(range(n)):
        u_prev = u_prev_first if k == 0 else controls[k - 1]
        c_val, lx, lu, lxx, luu, lux = cost["stage_derivs"](states[k], controls[k], u_prev, target)
        h_k = float(heading_traj[k]) if k < len(heading_traj) else 0.0
        _, fx, fu = dyn["step_with_jacobians"](states[k], controls[k], h_k, dt, params)

        q_x = lx + fx.T @ v_x
        q_u = lu + fu.T @ v_x
        q_xx = lxx + fx.T @ v_xx @ fx
        q_uu_raw = luu + fu.T @ v_xx @ fu
        q_uu = q_uu_raw + reg * I2
        q_ux = lux + fu.T @ v_xx @ fx

        q_uu_sym = 0.5 * (q_uu + q_uu.T)
        try:
            q_uu_inv = np.linalg.inv(q_uu_sym)
        except np.linalg.LinAlgError:
            return None
        # Rust solver bails on non-PD `q_uu`. Match the criterion.
        if np.trace(q_uu_inv) <= 0.0 or np.linalg.det(q_uu_sym) <= 0.0:
            return None

        k_vec = -q_uu_inv @ q_u
        kk = -q_uu_inv @ q_ux
        k_fb[k] = k_vec
        kk_fb[k] = kk

        dv1 += float(q_u @ k_vec)
        dv2 += 0.5 * float(k_vec @ q_uu_sym @ k_vec)

        v_x = q_x + q_ux.T @ k_vec
        v_xx = q_xx + q_ux.T @ kk
        v_xx = 0.5 * (v_xx + v_xx.T)

    return _Backward(k_fb=k_fb, kk_fb=kk_fb, expected_dv1=dv1, expected_dv2=dv2)


def _forward_pass(
    alpha: float,
    dyn: dict[str, Callable],
    cost: dict[str, Callable],
    x0: np.ndarray,
    prev_states: np.ndarray,
    prev_controls: np.ndarray,
    u_prev_first: np.ndarray,
    back: _Backward,
    heading_traj: np.ndarray,
    target: MpcTarget,
    params: RobotParams,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(prev_controls)
    new_states = np.empty((n + 1, 4))
    new_controls = np.empty((n, 2))
    new_states[0] = x0
    cost_total = 0.0
    u_prev = u_prev_first
    for k in range(n):
        dx = new_states[k] - prev_states[k]
        u_new = prev_controls[k] + alpha * back.k_fb[k] + back.kk_fb[k] @ dx
        new_controls[k] = u_new
        h_k = float(heading_traj[k]) if k < len(heading_traj) else 0.0
        cost_total += float(cost["stage_cost_scalar"](new_states[k], u_new, u_prev, target))
        new_states[k + 1] = dyn["step"](new_states[k], u_new, h_k, dt, params)
        u_prev = u_new
    return new_states, new_controls, cost_total


def _run_ilqr(
    dyn: dict[str, Callable],
    cost: dict[str, Callable],
    x0: np.ndarray,
    u_init: np.ndarray,
    u_prev_first: np.ndarray,
    heading_traj: np.ndarray,
    target: MpcTarget,
    params: RobotParams,
    cfg: SolverConfig,
) -> SolveResult:
    controls = u_init.copy()
    states, cost_val = _rollout(
        dyn["step"], cost["stage_cost_scalar"],
        x0, controls, u_prev_first, heading_traj, target, params, cfg.dt,
    )
    reg = max(cfg.reg_init, cfg.reg_min)
    converged = False
    iters = 0

    for it in range(cfg.max_iters):
        iters = it + 1
        # Backward pass with regularization growth on failure.
        back = None
        while True:
            back = _backward_pass(
                dyn, cost, states, controls, u_prev_first, heading_traj,
                target, params, cfg.dt, reg,
            )
            if back is not None:
                break
            reg = min(reg * cfg.reg_factor, cfg.reg_max)
            if reg >= cfg.reg_max:
                break
        if back is None:
            break

        accepted = False
        for alpha in _ALPHAS:
            ns, nc, new_cost = _forward_pass(
                alpha, dyn, cost, x0, states, controls, u_prev_first, back,
                heading_traj, target, params, cfg.dt,
            )
            expected = alpha * back.expected_dv1 + alpha * alpha * back.expected_dv2
            actual = new_cost - cost_val
            tol = 1.0e-8 * max(abs(cost_val), 1.0)
            if expected < -tol:
                accept = actual < 0.1 * expected
            else:
                accept = actual < -tol
            if accept:
                prev_cost = cost_val
                states, controls, cost_val = ns, nc, new_cost
                accepted = True
                reg = max(reg / cfg.reg_factor, cfg.reg_min)
                if abs(prev_cost - cost_val) < cfg.cost_tol * max(abs(prev_cost), 1.0):
                    converged = True
                break

        if not accepted:
            reg = min(reg * cfg.reg_factor, cfg.reg_max)
            if reg >= cfg.reg_max:
                break
        if converged:
            break

    return SolveResult(
        states=states, controls=controls, final_cost=float(cost_val),
        iters=iters, converged=converged,
    )


def solve(
    dyn: dict[str, Callable],
    cost: dict[str, Callable],
    x0: np.ndarray,
    heading_traj: np.ndarray,
    target: MpcTarget,
    params: RobotParams,
    warm_start: SolveResult | None = None,
    cfg: SolverConfig = SolverConfig(),
) -> SolveResult:
    """Multi-start iLQR: warm-start (shifted) + straight-line-to-target."""

    n = cfg.horizon
    x0 = np.asarray(x0, dtype=float).reshape(4)

    if warm_start is not None and len(warm_start.controls) > 0:
        u_prev_first = warm_start.controls[0].copy()
    else:
        u_prev_first = np.zeros(2)

    # Init 1: warm-start shifted by one stage, padded with the last control.
    if warm_start is not None and len(warm_start.controls) > 0:
        warm = np.empty((n, 2))
        last_idx = len(warm_start.controls) - 1
        for k in range(n):
            idx = min(k + 1, last_idx)
            warm[k] = warm_start.controls[idx]
    else:
        warm = np.zeros((n, 2))

    # Init 2: constant velocity that would reach the target in horizon·dt,
    # clipped to a sane cap to avoid wild initial controls.
    horizon_time = cfg.dt * n
    dir_vec = np.array([target.p.x - x0[0], target.p.y - x0[1]])
    if horizon_time > 1.0e-9:
        desired = dir_vec / horizon_time
    else:
        desired = np.zeros(2)
    SANE_CAP = 3500.0
    nrm = float(np.linalg.norm(desired))
    if nrm > SANE_CAP:
        desired = desired * (SANE_CAP / nrm)
    straight = np.tile(desired, (n, 1))

    r_warm = _run_ilqr(dyn, cost, x0, warm, u_prev_first, heading_traj, target, params, cfg)
    r_line = _run_ilqr(dyn, cost, x0, straight, u_prev_first, heading_traj, target, params, cfg)
    return r_warm if r_warm.final_cost <= r_line.final_cost else r_line
