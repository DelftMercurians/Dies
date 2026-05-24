"""Sanity tests for `pysim.lambdified.build` — wiring + numeric correctness."""

from __future__ import annotations

import numpy as np
import pytest

from formulations.cost import m as cost_model
from formulations.dynamics import m as dyn_model

from pysim import build, types


@pytest.fixture(scope="module")
def dyn():
    return build(dyn_model)


@pytest.fixture(scope="module")
def cost():
    return build(cost_model)


def _hand_step(x, u, heading, dt, p):
    """Hand-rolled forward-Euler dynamics matching the symbolic formulation."""
    c, s = np.cos(heading), np.sin(heading)
    R = np.array([[c, -s], [s, c]])
    v_global = x[2:]
    v_body = R.T @ v_global
    u_body = R.T @ u
    a_body = np.array([
        p.accel_max[0] * np.tanh((u_body[0] - v_body[0]) / (p.tau[0] * p.accel_max[0])),
        p.accel_max[1] * np.tanh((u_body[1] - v_body[1]) / (p.tau[1] * p.accel_max[1])),
    ])
    a_global = R @ a_body
    xdot = np.array([x[2], x[3], a_global[0], a_global[1]])
    return x + dt * xdot


def test_step_matches_hand_rolled(dyn):
    p = types.RobotParams.default_hand_tuned()
    samples = [
        (np.zeros(4), np.array([500.0, 0.0]), 0.0),
        (np.array([100.0, -50.0, 300.0, -200.0]), np.array([1500.0, 800.0]), 0.7),
        (np.array([-500.0, 400.0, -1200.0, 1500.0]), np.array([-2000.0, 1800.0]), -1.3),
    ]
    for x, u, h in samples:
        actual = dyn["step"](x, u, h, 0.06, p)
        expected = _hand_step(x, u, h, 0.06, p)
        np.testing.assert_allclose(actual, expected, atol=1e-9)


def test_step_with_jacobians_returns_tuple(dyn):
    p = types.RobotParams.default_hand_tuned()
    out = dyn["step_with_jacobians"](np.zeros(4), np.array([1000.0, 0.0]), 0.0, 0.06, p)
    assert isinstance(out, tuple) and len(out) == 3
    x_next, fx, fu = out
    assert x_next.shape == (4,)
    assert fx.shape == (4, 4)
    assert fu.shape == (4, 2)


def test_step_jacobians_match_finite_diff(dyn):
    p = types.RobotParams.default_hand_tuned()
    dt = 0.06
    eps = 1e-4
    samples = [
        (np.zeros(4), np.array([500.0, 0.0]), 0.0),
        (np.array([100.0, -50.0, 300.0, -200.0]), np.array([1500.0, 800.0]), 0.7),
        (np.array([-500.0, 400.0, -1200.0, 1500.0]), np.array([-2000.0, 1800.0]), -1.3),
    ]
    for x, u, h in samples:
        _, fx, fu = dyn["step_with_jacobians"](x, u, h, dt, p)
        for j in range(4):
            xp = x.copy(); xp[j] += eps
            xm = x.copy(); xm[j] -= eps
            fd = (dyn["step"](xp, u, h, dt, p) - dyn["step"](xm, u, h, dt, p)) / (2 * eps)
            np.testing.assert_allclose(fx[:, j], fd, atol=1e-6)
        for j in range(2):
            up = u.copy(); up[j] += eps
            um = u.copy(); um[j] -= eps
            fd = (dyn["step"](x, up, h, dt, p) - dyn["step"](x, um, h, dt, p)) / (2 * eps)
            np.testing.assert_allclose(fu[:, j], fd, atol=1e-6)


def test_stage_derivs_match_finite_diff(cost):
    target = types.MpcTarget(
        p=types.Vec2(1000.0, 500.0),
        v=types.Vec2(0.0, 0.0),
        weights=types.CostWeights(),
    )
    x = np.array([800.0, 200.0, -300.0, 150.0])
    u = np.array([450.0, -200.0])
    u_prev = np.array([100.0, 50.0])
    eps = 1e-4

    cost_val, lx, lu, lxx, luu, lux = cost["stage_derivs"](x, u, u_prev, target)
    assert isinstance(cost_val, float)
    assert lx.shape == (4,)
    assert lu.shape == (2,)
    assert lxx.shape == (4, 4)
    assert luu.shape == (2, 2)
    assert lux.shape == (2, 4)

    for j in range(4):
        xp = x.copy(); xp[j] += eps
        xm = x.copy(); xm[j] -= eps
        fd = (cost["stage_cost_scalar"](xp, u, u_prev, target)
              - cost["stage_cost_scalar"](xm, u, u_prev, target)) / (2 * eps)
        np.testing.assert_allclose(lx[j], fd, atol=1e-5)
    for j in range(2):
        up = u.copy(); up[j] += eps
        um = u.copy(); um[j] -= eps
        fd = (cost["stage_cost_scalar"](x, up, u_prev, target)
              - cost["stage_cost_scalar"](x, um, u_prev, target)) / (2 * eps)
        np.testing.assert_allclose(lu[j], fd, atol=1e-5)


def test_zero_at_target_with_zero_control(cost):
    target = types.MpcTarget(
        p=types.Vec2(1000.0, 500.0),
        v=types.Vec2(0.0, 0.0),
        weights=types.CostWeights(),
    )
    x = np.array([target.p.x, target.p.y, target.v.x, target.v.y])
    u = np.zeros(2)
    assert cost["stage_cost_scalar"](x, u, u, target) == pytest.approx(0.0, abs=1e-12)
