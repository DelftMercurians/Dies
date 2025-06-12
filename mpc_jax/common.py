from __future__ import annotations
import jax
from jax import numpy as jnp
import os
import equinox as eqx
from jaxtyping import Array, Float, Int

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
CONTROL_HORIZON = 20
TIME_HORIZON = 8  # 8 seconds
DT = 0.1  # Start with 0.1 seconds as DT
MAX_DT = 2 * TIME_HORIZON / CONTROL_HORIZON - DT  # Computed for linear dt schedule
ROBOT_RADIUS = 90.0  # mm
COLLISION_PENALTY_RADIUS = 200.0  # mm
FIELD_BOUNDARY_MARGIN = 100.0  # mm
MAX_ITERATIONS = 100
LEARNING_RATE = 100
UPDATE_CLIP = 50
N_CANDIDATE_TRAJECTORIES = 100

# Robot dynamics parameters
ROBOT_MASS = 1.5  # kg
VEL_FRICTION_COEFF = 0.5  # N*s/m (velocity-dependent friction coefficient)
MAX_ACC = 100  # i don't fucking know in what units this shit is


def get_dt_schedule() -> jax.Array:
    """Generate linearly increasing dt schedule from DT to MAX_DT"""
    steps = jnp.arange(CONTROL_HORIZON)
    return DT + (MAX_DT - DT) * steps / (CONTROL_HORIZON - 1)


Control = Float[Array, f"n_robots {CONTROL_HORIZON} 2"]


class Entity(eqx.Module):
    position: Float[Array, "2"]
    velocity: Float[Array, "2"]

    def __init__(self, pos, vel=None):
        self.position = pos
        self.velocity = jnp.zeros_like(self.position) if vel is None else vel

    def after(self, t: int | float | Float[Array, ""]):
        return Entity(self.position + self.velocity * t, self.velocity)


class EntityBatch(Entity):
    position: Float[Array, "n 2"]
    velocity: Float[Array, "n 2"]

    def __init__(self, pos, vel=None):
        # automatically wrap (2,) into (1,2)
        if len(pos.shape) == 1:
            pos = pos[None, :]
            vel = None if vel is None else vel[None, :]

        self.position = pos
        self.velocity = jnp.zeros_like(self.position) if vel is None else vel

    def get(self, idx: int | Int[Array, ""]):
        return Entity(self.position[idx], self.velocity[idx])

    def set(self, idx: int | Int[Array, ""], value: Entity):
        return EntityBatch(
            self.position.at[idx].set(value.position),
            self.velocity.at[idx].set(value.velocity),
        )

    def __len__(self):
        return len(self.position)

    def after(self, t: int | float | Float[Array, ""]):
        return EntityBatch(self.position + self.velocity * t, self.velocity)


class FieldBounds(eqx.Module):
    def bounding_box(self):
        return jnp.array([-2000.0, 2000.0, -1000.0, 1000.0])  # 4m x 2m field


class World(eqx.Module):
    """The current state of the world"""

    field_bounds: FieldBounds  # static constraints

    obstacles: EntityBatch  # obstacles (e.g. other robots)
    robots: EntityBatch
    ball: Entity = Entity(jnp.zeros((2,), dtype=jnp.float32))


@eqx.filter_jit
def trajectories_from_control(w: World, u: Control):
    return jax.vmap(single_trajectory_from_control)(
        u, w.robots.position, w.robots.velocity
    )


@jax.jit
def single_trajectory_from_control(
    control_sequence: jax.Array,
    initial_pos: jax.Array,
    initial_vel: jax.Array | None = None,
) -> jax.Array:
    if initial_vel is None:
        initial_vel = jnp.zeros(2)

    dt_schedule = get_dt_schedule()

    def dynamics(
        pos: jax.Array, vel: jax.Array, target_vel: jax.Array, dt: float
    ) -> jax.Array:
        vel_friction_force = -VEL_FRICTION_COEFF * vel
        control_force = (
            jnp.clip((target_vel - vel), -MAX_ACC, MAX_ACC) * ROBOT_MASS / dt
        )

        # Total acceleration
        total_force = control_force + vel_friction_force
        acceleration = total_force / ROBOT_MASS
        return acceleration

    def heun_step(state: jax.Array, target_vel: jax.Array, dt: float) -> jax.Array:
        """Heun method for state integration"""
        pos, vel = state[:2], state[2:]

        # First evaluation
        acc1 = dynamics(pos, vel, target_vel, dt)

        # Predictor step
        pos_pred = pos + vel * dt
        vel_pred = vel + acc1 * dt

        # Second evaluation
        acc2 = dynamics(pos_pred, vel_pred, target_vel, dt)

        # Corrector step
        new_pos = pos + vel * dt + 0.5 * acc1 * dt * dt
        new_vel = vel + 0.5 * (acc1 + acc2) * dt

        return jnp.concatenate([new_pos, new_vel])

    def scan_fn(carry, inputs):
        control, dt = inputs
        state, time = carry
        new_state = heun_step(state, control, dt)
        new_time = time + dt
        return (new_state, new_time), jnp.concatenate([new_time[None], new_state])

    # Initial state includes position and velocity
    initial_state_vec = jnp.concatenate([initial_pos, initial_vel])

    # Use scan to efficiently compute trajectory with time
    inputs = (control_sequence, dt_schedule)
    _, trajectory_steps = jax.lax.scan(scan_fn, (initial_state_vec, 0.0), inputs)

    # Create initial state with time=0
    initial_state = jnp.concatenate([jnp.array([0.0]), initial_pos, initial_vel])

    # Prepend initial state to get full trajectory
    return jnp.concatenate([initial_state[None, :], trajectory_steps], axis=0)
