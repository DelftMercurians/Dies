"""
JAX-based Model Predictive Control (MPC) for robot navigation.

This package provides a JAX implementation of MPC for real-time robot path planning
and obstacle avoidance using automatic differentiation and JIT compilation.
"""

import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax

jax.config.update("jax_compilation_cache_dir", ".jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

# jax.config.update("jax_disable_jit", True)
jax.config.update("jax_log_compiles", True)
# jax.config.update("jax_debug_nans", True)

from .main import solve_mpc, solve_mpc_tbwrap

__version__ = "0.1.0"
__all__ = ["solve_mpc", "solve_mpc_tbwrap"]
