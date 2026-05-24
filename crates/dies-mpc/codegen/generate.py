#!/usr/bin/env python3
"""Generate analytic Rust code for dies-mpc dynamics and cost derivatives.

Regenerate with:
    uv run --project crates/dies-mpc/codegen python crates/dies-mpc/codegen/generate.py

Check (CI-style staleness diff, exit 1 if out of date):
    uv run --project crates/dies-mpc/codegen python crates/dies-mpc/codegen/generate.py --check
"""

from formulations import cost, dynamics
from ilqr_codegen.io import run

if __name__ == "__main__":
    raise SystemExit(run([dynamics.m, cost.m]))
