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
CONTROL_HORIZON = 10
TIME_HORIZON = 5  # seconds
DT = 0.05  # starting value for dt, seconds
MAX_DT = 2 * TIME_HORIZON / CONTROL_HORIZON - DT  # Computed for linear dt schedule
ROBOT_RADIUS = 90.0  # mm
COLLISION_PENALTY_RADIUS = 200.0  # mm
FIELD_BOUNDARY_MARGIN = 100.0  # mm
MAX_ITERATIONS = 500
LEARNING_RATE = 50
N_CANDIDATE_TRAJECTORIES = 40
TRAJECTORY_RESOLUTION = 6  # points per physics step for high-resolution trajectories

# Robot dynamics parameters
ROBOT_MASS = 1.5  # kg
VEL_FRICTION_COEFF = 1e-10  # N*s/m (velocity-dependent friction coefficient)
MAX_ACC = 10_000  # i don't fucking know in what units this shit is


def get_dt_schedule(upscaled=True):
    """Generate linearly increasing dt schedule from DT to MAX_DT"""
    if upscaled:
        # The schedule that follows interpolation - linear interpolation of cumulative times
        base_dt_schedule = get_dt_schedule(upscaled=False)
        base_cumulative_times = jnp.concatenate(
            [jnp.array([0.0]), jnp.cumsum(base_dt_schedule)]
        )

        # Total interpolated points: CONTROL_HORIZON + (CONTROL_HORIZON-1) * (TRAJECTORY_RESOLUTION-1)
        total_points = CONTROL_HORIZON + (CONTROL_HORIZON - 1) * (
            TRAJECTORY_RESOLUTION - 1
        )

        # Linear interpolation from 0 to final time
        final_time = base_cumulative_times[-1]
        interpolated_times = jnp.linspace(0, final_time, total_points)

        # Convert to dt schedule by taking differences
        return jnp.diff(interpolated_times)
    else:
        steps = jnp.arange(CONTROL_HORIZON)
        return DT + (MAX_DT - DT) * steps / (CONTROL_HORIZON - 1)


def control_steps_to_time(time: float | Float[Array, ""]):
    """Return the smallest number of steps such that the control will be past this time point"""
    dt_schedule = get_dt_schedule(upscaled=False)
    cumulative_time = jnp.cumsum(dt_schedule)
    steps_past_time = jnp.searchsorted(cumulative_time, time, side="right")
    return steps_past_time


@jax.jit
def cubic_hermite_spline(
    t: jax.Array, p0: jax.Array, p1: jax.Array, v0: jax.Array, v1: jax.Array, dt: float
) -> tuple:
    """Cubic Hermite spline interpolation between two points with velocities"""
    # Hermite basis functions
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2

    # Interpolate position
    pos = (
        h00[..., None] * p0
        + h10[..., None] * (dt * v0)
        + h01[..., None] * p1
        + h11[..., None] * (dt * v1)
    )

    # Interpolate velocity (derivative of position)
    h00_dot = 6 * t**2 - 6 * t
    h10_dot = 3 * t**2 - 4 * t + 1
    h01_dot = -6 * t**2 + 6 * t
    h11_dot = 3 * t**2 - 2 * t

    vel = (
        h00_dot[..., None] * p0
        + h10_dot[..., None] * (dt * v0)
        + h01_dot[..., None] * p1
        + h11_dot[..., None] * (dt * v1)
    ) / dt

    return pos, vel


@eqx.filter_jit
def interpolate_trajectory_segment(
    t0: float, t1: float, state0: jax.Array, state1: jax.Array, n_points: int
) -> jax.Array:
    """Interpolate between two trajectory points using cubic Hermite spline"""
    dt = t1 - t0
    p0, v0 = state0[:2], state0[2:]
    p1, v1 = state1[:2], state1[2:]

    # Create interpolation parameters (excluding endpoints to avoid duplication)
    t_params = jnp.linspace(0, 1, n_points + 2)[1:-1]  # exclude 0 and 1

    # Interpolate positions and velocities
    pos_interp, vel_interp = cubic_hermite_spline(t_params, p0, p1, v0, v1, dt)

    # Create time points
    time_interp = t0 + t_params * dt

    # Combine into trajectory format [time, pos_x, pos_y, vel_x, vel_y]
    return jnp.column_stack([time_interp, pos_interp, vel_interp])


@jax.jit
def spline_interpolate_trajectory(trajectory: jax.Array) -> jax.Array:
    """Add high-resolution interpolated points between trajectory segments"""
    resolution = TRAJECTORY_RESOLUTION
    if resolution <= 1:
        return trajectory

    n_segments = len(trajectory) - 1

    def interpolate_segment(i):
        t0, t1 = trajectory[i, 0], trajectory[i + 1, 0]
        state0, state1 = trajectory[i, 1:], trajectory[i + 1, 1:]
        return interpolate_trajectory_segment(t0, t1, state0, state1, resolution - 1)

    # Interpolate all segments
    interpolated_segments = jax.vmap(interpolate_segment)(jnp.arange(n_segments))

    # Flatten interpolated segments and combine with original points
    interpolated_points = interpolated_segments.reshape(-1, 5)

    # Create full high-resolution trajectory by interleaving original and interpolated points
    result_parts = []
    for i in range(n_segments):
        result_parts.append(trajectory[i : i + 1])  # original point
        start_idx = i * (resolution - 1)
        end_idx = (i + 1) * (resolution - 1)
        result_parts.append(
            interpolated_points[start_idx:end_idx]
        )  # interpolated points

    # Add final point
    result_parts.append(trajectory[-1:])

    return jnp.concatenate(result_parts, axis=0)


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
    return jax.vmap(
        lambda control, pos, vel: single_trajectory_from_control(control, pos, vel)
    )(u, w.robots.position, w.robots.velocity)


@jax.jit
def single_trajectory_from_control(
    control_sequence: jax.Array,
    initial_pos: jax.Array,
    initial_vel: jax.Array | None = None,
) -> jax.Array:
    if initial_vel is None:
        initial_vel = jnp.zeros(2)

    dt_schedule = get_dt_schedule(upscaled=False)

    def dynamics(
        pos: jax.Array, vel: jax.Array, target_vel: jax.Array, dt: float
    ) -> jax.Array:
        vel_friction_force = -VEL_FRICTION_COEFF * vel
        control_force = (
            jnp.clip((target_vel - vel) / dt, -MAX_ACC, MAX_ACC) * ROBOT_MASS
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
    base_trajectory = jnp.concatenate(
        [initial_state[None, :], trajectory_steps], axis=0
    )

    # Apply spline interpolation for higher resolution
    return spline_interpolate_trajectory(base_trajectory)
