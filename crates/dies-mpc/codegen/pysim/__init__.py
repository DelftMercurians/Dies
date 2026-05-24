"""Python sandbox for tuning dies-mpc dynamics and cost.

Shares the symbolic formulations with Rust codegen — same `Model` objects
that drive `generate.py` are lambdified into NumPy callables here.

Typical use from a notebook:

    import numpy as np
    from formulations.dynamics import m as dyn_model
    from formulations.cost     import m as cost_model
    from pysim import build, solve, rollout, cost_along, types

    dyn  = build(dyn_model)
    cost = build(cost_model)
    p = types.RobotParams.default_hand_tuned()
    target = types.MpcTarget.goto(types.Vec2(2000.0, 0.0))
    result = solve(dyn, cost, np.zeros(4), np.zeros(151), target, p)
"""

from . import types
from .ilqr import solve
from .lambdified import build
from .simulator import cost_along, rollout

__all__ = ["build", "rollout", "cost_along", "solve", "types"]
