"""
JAX-based Model Predictive Control (MPC) for robot navigation.

This package provides a JAX implementation of MPC for real-time robot path planning
and obstacle avoidance using automatic differentiation and JIT compilation.
"""

from .main import solve_mpc, mpc_cost_function, test_simple_case

__version__ = "0.1.0"
__all__ = ["solve_mpc", "mpc_cost_function", "test_simple_case"]