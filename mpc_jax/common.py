import jax
from jax import numpy as jnp
import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)

# jax.config.update("jax_disable_jit", True)
# jax.config.update("jax_log_compiles", True)
jax.config.update("jax_debug_nans", True)

# MPC Parameters
PREDICTION_HORIZON = 30
DT = 0.2
ROBOT_RADIUS = 90.0  # mm
COLLISION_PENALTY_RADIUS = 200.0  # mm
FIELD_BOUNDARY_MARGIN = 100.0  # mm
MAX_ITERATIONS = 20
LEARNING_RATE = 400
UPDATE_CLIP = 200

# Robot dynamics parameters
ROBOT_MASS = 1.5  # kg
VEL_FRICTION_COEFF = 0.5  # N*s/m (velocity-dependent friction coefficient)
MAX_ACC = 100


@jax.jit
def trajectory_from_control(
    control_sequence: jax.Array,
    initial_pos: jax.Array,
    initial_vel: jax.Array = None,
) -> jax.Array:
    if initial_vel is None:
        initial_vel = jax.numpy.zeros(2)
    dt = DT

    def dynamics(pos: jax.Array, vel: jax.Array, target_vel: jax.Array) -> jax.Array:
        vel_friction_force = -VEL_FRICTION_COEFF * vel
        control_force = (
            jnp.clip((target_vel - vel), -MAX_ACC, MAX_ACC) * ROBOT_MASS / dt
        )

        # Total acceleration
        total_force = control_force + vel_friction_force
        acceleration = total_force / ROBOT_MASS
        return acceleration

    def heun_step(state: jax.Array, target_vel: jax.Array) -> jax.Array:
        """Heun method for state integration"""
        pos, vel = state[:2], state[2:]

        # First evaluation
        acc1 = dynamics(pos, vel, target_vel)

        # Predictor step
        pos_pred = pos + vel * dt
        vel_pred = vel + acc1 * dt

        # Second evaluation
        acc2 = dynamics(pos_pred, vel_pred, target_vel)

        # Corrector step
        new_pos = pos + vel * dt + 0.5 * acc1 * dt * dt
        new_vel = vel + 0.5 * (acc1 + acc2) * dt

        return jax.numpy.concatenate([new_pos, new_vel])

    def scan_fn(carry, control):
        state, time = carry
        new_state = heun_step(state, control)
        new_time = time + dt
        return (new_state, new_time), jax.numpy.concatenate([new_time[None], new_state])

    # Initial state includes position and velocity
    initial_state_vec = jax.numpy.concatenate([initial_pos, initial_vel])

    # Use scan to efficiently compute trajectory with time
    _, trajectory_steps = jax.lax.scan(
        scan_fn, (initial_state_vec, 0.0), control_sequence
    )

    # Create initial state with time=0
    initial_state = jax.numpy.concatenate(
        [jax.numpy.array([0.0]), initial_pos, initial_vel]
    )

    # Prepend initial state to get full trajectory
    return jax.numpy.concatenate([initial_state[None, :], trajectory_steps], axis=0)
