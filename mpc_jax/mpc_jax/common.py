from __future__ import annotations
import jax
from jax import numpy as jnp
from jax import random as jr
import os
import equinox as eqx
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Literal


# MPC Parameters
CONTROL_HORIZON = 8
TIME_HORIZON = 2  # seconds
DT = 0.04  # starting value for dt, seconds
MAX_DT = 2 * TIME_HORIZON / CONTROL_HORIZON - DT  # Computed for linear dt schedule
ROBOT_RADIUS = 90.0  # mm
BALL_RADIUS = 21.35  # mm
COLLISION_PENALTY_RADIUS = 200.0  # mm
FIELD_BOUNDARY_MARGIN = 100.0  # mm
MAX_ITERATIONS = 50
BATCH_SIZE = 5
LEARNING_RATE = 30
N_CANDIDATE_TRAJECTORIES = 10
TRAJECTORY_RESOLUTION = 8  # points per physics step for high-resolution trajectories
FINAL_COST: Literal["distance-auc", "cost"] = "distance-auc"

# Robot dynamics parameters
ROBOT_MASS = 1.5  # kg
VEL_FRICTION_COEFF = 0  # N*s/m (velocity-dependent friction coefficient)
MAX_ACC = 12_000  # mm/s^2


def add_control_noise(
    key: PRNGKeyArray,
    control: Control,
) -> Control:
    k1, k2, k3 = jr.split(key, 3)
    noise = jr.normal(k1, control.shape) * 5.0
    scale = jr.uniform(k2, (len(control),), minval=1.0, maxval=1.02)
    uni_scale = jr.uniform(k3, (1,), minval=1.0, maxval=1.1)
    return control * scale[:, None, None] * uni_scale[None, None, :] + noise


class MPCConfig(eqx.Module):
    distance_factor: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(1.0))
    collision_factor: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(1.0))
    ball_collision_factor: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(1.0)
    )
    ball_min_safe_distance: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(ROBOT_RADIUS * 1.05 + BALL_RADIUS)
    )
    ball_no_cost_distance: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(ROBOT_RADIUS * 2 + BALL_RADIUS)
    )
    obstacle_min_safe_distance: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(ROBOT_RADIUS * 2.2)
    )
    obstacle_no_cost_distance: jax.Array = eqx.field(
        default_factory=lambda: jnp.asarray(ROBOT_RADIUS * 4)
    )
    delay: jax.Array = eqx.field(default_factory=lambda: jnp.asarray(0.0))

    @staticmethod
    def sample(key):
        return MPCConfig()

    def noisy(self, key, noise_scale: float = 0.0):
        """Add noise to delay parameter."""
        noise = jax.random.normal(key, ()) * noise_scale
        return eqx.tree_at(
            lambda cfg: cfg.delay, self, jnp.maximum(0.0, self.delay + noise)
        )


class Result(eqx.Module):
    u: jax.Array
    traj: jax.Array
    cost: jax.Array
    idx_by_cost: jax.Array


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


def softclip(x: jax.Array, max_norm: float) -> jax.Array:
    alpha = 2
    if x.ndim == 1 and len(x) > 1:
        norm = jnp.sqrt((x**2).sum() + 1e-12)

        scale = max_norm / norm
        scale = jnp.minimum(scale, 1.0)  # safety for tiny norm

        return x * scale
    else:
        raise RuntimeError("blah blah")


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
    mask: Int[Array, "n"]

    def __init__(self, pos, vel=None, n=6):
        # automatically wrap (2,) into (1,2)
        n = max(n, pos.size // 2)
        if len(pos.shape) == 1:
            pos = pos[None, :]
            vel = None if vel is None else vel[None, :]

        self.position = jnp.zeros((n, 2)).at[: len(pos)].set(pos)
        vel = jnp.zeros_like(self.position) if vel is None else vel
        self.velocity = jnp.zeros((n, 2)).at[: len(vel)].set(vel)
        self.mask = jnp.arange(n) < len(pos)

    def get(self, idx: int | Int[Array, ""]):
        return Entity(self.position[idx], self.velocity[idx])

    def set(self, idx: int | Int[Array, ""], value: Entity):
        return EntityBatch(
            self.position.at[idx].set(value.position),
            self.velocity.at[idx].set(value.velocity),
        )

    def dyn_len(self):
        return jnp.count_nonzero(self.mask)

    def st_len(self):
        return len(self.mask)

    def after(self, t: int | float | Float[Array, ""]):
        return eqx.tree_at(
            lambda s: s.position, self, self.position + self.velocity * t
        )


class FieldBounds(eqx.Module):
    field_length: float = 4000.0  # mm
    field_width: float = 2000.0  # mm
    penalty_area_depth: float = 1000.0  # mm
    penalty_area_width: float = 2000.0  # mm

    def __init__(
        self,
        field_length: float = 4000.0,
        field_width: float = 2000.0,
        penalty_area_depth: float = 1000.0,
        penalty_area_width: float = 2000.0,
    ):
        self.field_length = field_length
        self.field_width = field_width
        self.penalty_area_depth = penalty_area_depth
        self.penalty_area_width = penalty_area_width

    def bounding_box(self):
        half_length = self.field_length / 2.0
        half_width = self.field_width / 2.0
        return jnp.array([-half_length, half_length, -half_width, half_width])


class World(eqx.Module):
    """The current state of the world"""

    field_bounds: FieldBounds  # static constraints

    obstacles: EntityBatch  # obstacles (e.g. other robots)
    robots: EntityBatch
    ball: Entity = Entity(jnp.zeros((2,), dtype=jnp.float32))
    avoid_goal_area: bool = False

    def noisy(self, key, pos_noise_scale: float = 30.0, vel_noise_scale: float = 10.0):
        """Add noise to robot positions and velocities."""
        pos_vel_keys = jax.random.split(key, self.robots.st_len())

        noisy_positions = []
        noisy_velocities = []

        for i, pos_vel_key in enumerate(pos_vel_keys):
            k1, k2 = jax.random.split(pos_vel_key)
            pos_noise = jax.random.normal(k1, (2,)) * pos_noise_scale
            vel_noise = jax.random.normal(k2, (2,)) * vel_noise_scale

            noisy_positions.append(self.robots.position[i] + pos_noise)
            noisy_velocities.append(self.robots.velocity[i] + vel_noise)

        noisy_positions = jnp.stack(noisy_positions)
        noisy_velocities = jnp.stack(noisy_velocities)

        return eqx.tree_at(
            lambda w: w.robots, self, EntityBatch(noisy_positions, noisy_velocities)
        )


def trajectories_from_control(w: World, u: Control, delay: float = 0.0):
    return jax.vmap(
        lambda control, pos, vel: single_trajectory_from_control(
            control, pos, vel, delay
        )
    )(u, w.robots.position, w.robots.velocity)


def clip_by_norm(vel: jnp.ndarray, limit: float | Float[Array, ""]):
    speed = jnp.sqrt((vel**2).sum() + 1e-9)
    return jnp.where(speed > limit, vel * (limit / speed), vel)


def single_trajectory_from_control(
    control_sequence: jax.Array,
    initial_pos: jax.Array,
    initial_vel: jax.Array,
    delay: float = 0.0,
) -> jax.Array:
    initial_pos = initial_pos + initial_vel * 0.1

    dt_schedule = get_dt_schedule(upscaled=False)

    def dynamics(
        pos: jax.Array, vel: jax.Array, target_vel: jax.Array, dt: float
    ) -> jax.Array:
        vel_friction_force = -VEL_FRICTION_COEFF * vel

        desired_acceleration = (target_vel - vel) / dt

        clipped_acceleration = clip_by_norm(desired_acceleration, MAX_ACC)

        control_force = clipped_acceleration * ROBOT_MASS

        total_force = control_force + vel_friction_force
        return total_force / ROBOT_MASS

    def heun_step(state: jax.Array, target_vel: jax.Array, dt: float) -> jax.Array:
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

    initial_state = jnp.concatenate([jnp.array([delay]), initial_pos, initial_vel])

    inputs = (control_sequence, dt_schedule)
    _, trajectory_steps = jax.lax.scan(scan_fn, (initial_state[1:], delay), inputs)

    base_trajectory = jnp.concatenate(
        [initial_state[None, :], trajectory_steps], axis=0
    )

    # Apply spline interpolation for higher resolution
    return spline_interpolate_trajectory(base_trajectory)
