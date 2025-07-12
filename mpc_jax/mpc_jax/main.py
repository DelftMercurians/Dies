import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, value_and_grad
from jaxtyping import PRNGKeyArray, Float, Array
import numpy as np
import traceback as tb
import optax
import os
import warnings
import equinox as eqx
import time
from tqdm import tqdm
import functools as ft
import matplotlib.pyplot as plt
from .common import (
    BALL_RADIUS,
    FINAL_COST,
    ROBOT_RADIUS,
    BATCH_SIZE,
    LEARNING_RATE,
    CONTROL_HORIZON,
    MAX_ITERATIONS,
    TIME_HORIZON,
    N_CANDIDATE_TRAJECTORIES,
    add_control_noise,
    single_trajectory_from_control,
    trajectories_from_control,
    World,
    Control,
    Entity,
    EntityBatch,
    FieldBounds,
    control_steps_to_time,
    clip_by_norm,
    Result,
    MPCConfig,
    TRAJECTORY_RESOLUTION,
    get_dt_schedule,
)

from .costs import (
    distance_cost,
    collision_cost,
    ball_collision_cost,
    boundary_cost,
    velocity_constraint_cost,
)

from .scoring import select_best_stochastic_trajectory


def mpc_cost_function(
    u: Control,
    w: World,
    targets: EntityBatch,
    max_speeds: jax.Array,
    cfg,
    key,
):
    k1, k2, k3 = jr.split(key, 3)
    w = w.noisy(k1)
    cfg = cfg.noisy(k2)
    u = add_control_noise(k3, u)
    trajectories = trajectories_from_control(w, u, cfg.delay)

    # Compute position-based costs over the trajectory
    def position_cost_fn(raw_traj, max_speed, target):
        t, pos_x, pos_y, vel_x, vel_y = raw_traj
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        # Account for delay in target position prediction
        delayed_target_pos = target.after(t + cfg.delay).position
        d_cost = distance_cost(robot.position, delayed_target_pos, t)
        # collision with obstacles (non-controllable robots) costs
        c_cost = 0
        if w.obstacles.st_len() != 0:
            # Account for delay in obstacle position prediction
            delayed_obstacle_pos = w.obstacles.after(t + cfg.delay).position
            c_cost = collision_cost(
                robot.position,
                delayed_obstacle_pos,
                mask=w.obstacles.mask,
                min_safe_distance=cfg.obstacle_min_safe_distance,
                no_cost_distance=cfg.obstacle_no_cost_distance,
            )
        # Handle ball collision separately with proper radius
        delayed_ball_pos = w.ball.after(t + cfg.delay).position
        ball_cost = ball_collision_cost(
            robot.position,
            delayed_ball_pos,
            min_safe_distance=cfg.ball_min_safe_distance,
            no_cost_distance=cfg.ball_no_cost_distance,
        )

        # other stuff
        b_cost = boundary_cost(robot.position, w.field_bounds)
        vc_cost = velocity_constraint_cost(robot.velocity, max_speed)
        dist = jnp.sqrt(((delayed_target_pos - robot.position) ** 2 + 1e-9).sum())
        effort_cost = jnp.sqrt((u**2).sum() + 1e-8) * 1e-4 / dist
        # TODO: increase the cost drastically if we are past the limit
        acc_cost = ((u[:, 1:, :] - u[:, :1, :]) ** 2).sum() * 3e-10

        return (
            d_cost * cfg.distance_factor
            + c_cost * cfg.collision_factor
            + ball_cost * cfg.ball_collision_factor
            + b_cost
            + vc_cost
            + effort_cost
            + acc_cost
        ) * target.mask

    def collective_position_cost_fn(traj_slice, idx, is_bad_robot):
        # collisions with our own robots cost
        t, pos_x, pos_y, vel_x, vel_y = traj_slice[idx]
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        obstacles = jax.lax.stop_gradient(traj_slice[:, 1:3])
        ours_c_cost = collision_cost(
            robot.position,
            obstacles,
            min_safe_distance=cfg.obstacle_min_safe_distance,
            no_cost_distance=cfg.obstacle_no_cost_distance,
        )
        return jax.lax.cond(is_bad_robot, lambda: 0.0, lambda: ours_c_cost * 2)

    # Skip initial position (i=0) and compute costs for trajectory steps
    total_position_cost = jax.vmap(
        lambda traj, target, max_speed: jax.vmap(
            eqx.Partial(position_cost_fn, max_speed=max_speed, target=target)
        )(traj)
    )(trajectories, targets, max_speeds)

    total_collective_cost = jax.vmap(
        lambda traj_slice: jax.vmap(
            eqx.Partial(collective_position_cost_fn, traj_slice)
        )(jnp.arange(len(w.robots.mask)), w.robots.mask),
        in_axes=1,
    )(trajectories).reshape(w.robots.st_len(), -1)

    tie_breaker_cost = 0

    return (
        (total_position_cost + total_collective_cost + tie_breaker_cost)
        * w.robots.mask[:, None]
    ).sum()


def generate_candidate_control(
    key: PRNGKeyArray,
    initial_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    max_speed: float,
) -> Float[Array, f"{CONTROL_HORIZON} 2"]:
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    # Generate pseudo-target with normal noise (std 50cm = 500mm)
    pos_noise_scale = jnp.clip(
        (jnp.linalg.norm(initial_pos - target_pos) - 30) / 400, 0, 1
    )
    noise = jr.normal(k1, (2,)) * 500.0 * pos_noise_scale
    pseudo_target = target_pos + noise

    # Stage 1: Go to 75%ish of distance
    ratio = jr.uniform(k6, (), minval=0.3, maxval=1.2)
    intermediate_target = initial_pos + ratio * (pseudo_target - initial_pos)

    # Calculate required velocity to reach 75% in a few seconds
    few_seconds = jr.uniform(k5, (), minval=2.0, maxval=8.0)
    required_vel_stage1 = (intermediate_target - initial_pos) / few_seconds
    required_vel_stage1 = clip_by_norm(required_vel_stage1, max_speed)

    # Add noise to stage 1 velocity
    noise1 = jr.normal(k2, (2,)) * 100.0 * pos_noise_scale
    first_stage_random_vel_clip_factor = jr.uniform(k3, (), minval=0.2, maxval=1.0)
    stage1_vel = clip_by_norm(
        required_vel_stage1 + noise1, max_speed * first_stage_random_vel_clip_factor
    )

    # Stage 2: Go from intermediate to full pseudo-target
    remaining_distance = jnp.linalg.norm(pseudo_target - intermediate_target)
    required_vel_stage2 = (pseudo_target - intermediate_target) / (
        TIME_HORIZON - few_seconds
    )
    required_vel_stage2 = clip_by_norm(required_vel_stage2, max_speed)

    # Add noise to stage 2 velocity
    noise2 = jr.normal(k4, (2,)) * 30.0
    stage2_vel = clip_by_norm(required_vel_stage2 + noise2, max_speed)

    # Combine the two stages using masks for JIT compatibility
    stage1_steps = control_steps_to_time(few_seconds)

    # Create mask for stage1 vs stage2
    time_indices = jnp.arange(CONTROL_HORIZON)
    stage1_mask = time_indices < stage1_steps

    # Use where to select between stage1 and stage2 velocities
    control_sequence = jnp.where(
        stage1_mask[:, None], stage1_vel[None, :], stage2_vel[None, :]
    )

    return control_sequence


def generate_continuity_candidates(
    key: PRNGKeyArray,
    last_control_sequences: jax.Array,
    max_speeds: jax.Array,
    n_robots: int,
) -> Float[Array, f"n_robots {CONTROL_HORIZON} 2"]:
    k1, k2 = jr.split(key, 2)

    # Add Gaussian noise to the last control sequences
    noise_scale = 100.0  # mm/s noise
    noise = jr.normal(k1, last_control_sequences.shape) * noise_scale

    # Apply noise with time-decay (less noise for earlier time steps)
    time_decay = jnp.linspace(1.0, 0.1, CONTROL_HORIZON)
    noise = noise * time_decay[None, :, None]

    perturbed_controls = last_control_sequences + noise

    # Clip to velocity limits
    perturbed_controls = jax.vmap(
        lambda controls, max_speed: jax.vmap(
            lambda control: clip_by_norm(control, max_speed)
        )(controls)
    )(perturbed_controls, max_speeds)

    return perturbed_controls


def solve_mpc_jax(
    w: World,
    targets: EntityBatch,
    max_speeds: jax.Array,
    last_control_sequences: jax.Array,
    max_iterations: int = MAX_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    n_candidates: int = N_CANDIDATE_TRAJECTORIES,
) -> Result:
    key = jr.key(0)
    cfg = MPCConfig()
    n_robots = w.robots.st_len()

    # Generate candidate trajectories - mix of random and continuity-based
    continuity_candidates = (
        int(n_candidates * 0.2) if last_control_sequences is not None else 0
    )
    random_candidates = n_candidates - continuity_candidates
    config_key, noise_key, key = jr.split(key, 3)

    keys = jr.split(key, n_candidates)

    # Generate random candidates
    random_keys = keys[:random_candidates]
    random_candidate_controls = jax.vmap(
        lambda k: jax.vmap(eqx.Partial(generate_candidate_control))(
            jr.split(k, n_robots),
            w.robots.position,
            targets.after(3).position,  # this is a heuristic, since why not
            max_speeds.reshape((n_robots, 1)),
        )
    )(random_keys)

    # Generate continuity-based candidates (perturbations of last control)
    continuity_keys = keys[random_candidates:]
    continuity_candidate_controls = jax.vmap(
        lambda k: generate_continuity_candidates(
            k, last_control_sequences, max_speeds, n_robots
        )
    )(continuity_keys)

    # Combine random and continuity candidates
    candidate_controls = jnp.concatenate(
        [random_candidate_controls, continuity_candidate_controls], axis=0
    )
    candidate_controls = candidate_controls.reshape(
        (n_candidates, n_robots, CONTROL_HORIZON, 2)
    )

    def batched_cost(u, key):
        return eqx.filter_vmap(
            lambda k: mpc_cost_function(
                u=u,
                w=w,
                targets=targets,
                max_speeds=max_speeds,
                cfg=cfg,
                key=k,
            )
        )(jr.split(key, BATCH_SIZE)).mean()

    # Optimize each candidate trajectory via (full-batch) gradient descent
    def optimize_control(u, key, cfg: MPCConfig = MPCConfig()):
        # Initialize optimizer for this trajectory
        lr_schedule = optax.linear_schedule(
            learning_rate, learning_rate / 10.0, max_iterations
        )
        optimizer = optax.chain(
            optax.adabelief(learning_rate=lr_schedule, b1=0.9, b2=0.9),
        )
        opt_state = optimizer.init(u)

        def optimization_step(carry, _):
            u, opt_state, (best_u, best_cost), key = carry
            key, subkey = jr.split(key)

            # Compute gradient wrt complete cost function
            cost, grad_val = eqx.filter_value_and_grad(batched_cost)(u, subkey)
            best_u = jnp.where(cost < best_cost, u, best_u)
            best_cost = jnp.where(cost < best_cost, cost, best_cost)

            # Do an optimization step
            updates, opt_state = optimizer.update(grad_val, opt_state, u)
            u: jax.Array = optax.apply_updates(u, updates)  # type: ignore

            # Project the control back to the "valid" domain
            u = jax.vmap(clip_by_norm)(u, max_speeds)

            return (u, opt_state, (best_u, best_cost), key), cost

        (_, _, (best_u, best_cost), _), _ = jax.lax.scan(
            optimization_step,
            (u, opt_state, (u, jnp.inf), key),
            None,
            length=max_iterations,
        )

        return best_u, best_cost

    # Optimize all candidate controls in parallel
    configs = eqx.filter_vmap(MPCConfig.sample)(
        jr.split(config_key, len(candidate_controls))
    )
    optimized_controls, mpc_costs = jax.vmap(optimize_control)(
        candidate_controls, jr.split(noise_key, len(candidate_controls)), configs
    )

    # Select best collision-free trajectory based on distance integral
    sind = jnp.argsort(mpc_costs)
    candidate_controls = candidate_controls[sind]
    optimized_controls = optimized_controls[sind]
    mpc_costs = mpc_costs[sind]

    best_mpc_idx = jnp.argmin(mpc_costs)
    best_mpc_cost = mpc_costs[best_mpc_idx]
    if FINAL_COST == "cost":
        best_idx = best_mpc_idx
        best_cost = best_mpc_cost
    elif FINAL_COST == "distance-auc":
        # Use stochastic trajectory scoring instead of deterministic collision checking
        stochastic_key, key = jr.split(key)
        best_stoch_idx, best_stoch_cost = select_best_stochastic_trajectory(
            stochastic_key, optimized_controls, w, targets, last_control_sequences, cfg
        )
        best_idx = jnp.where(best_stoch_cost == jnp.inf, best_mpc_idx, best_stoch_idx)
        best_cost = best_stoch_cost
    else:
        raise ValueError(
            f"FINAL_COST was {FINAL_COST}, but must be either of ['cost', 'distance-auc']"
        )

    u = optimized_controls[best_idx]
    traj = jax.vmap(
        lambda control, pos, vel: single_trajectory_from_control(
            control, pos, vel, cfg.delay
        )
    )(u, w.robots.position, w.robots.velocity)

    return Result(
        u=u,
        traj=traj,
        cost=best_cost,
        idx_by_cost=best_idx,
    )


def pprint(arr):
    out = "["
    for v in arr.ravel():
        out += f"{float(v):.2f} "
    return out[:-1] + "]" if len(out) != 1 else "[]"


_jitted = eqx.filter_jit(solve_mpc_jax)


def solve_mpc(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    ball_pos: np.ndarray,
    field_bounds: np.ndarray | None,
    max_speeds: np.ndarray,
    last_control_sequences: np.ndarray | None = None,
    dt: np.ndarray | None = np.array(0.02),
    field_geometry: np.ndarray | None = None,
    controllable_mask: np.ndarray | None = None,
) -> Result:
    n_robots = len(initial_pos)
    ctrl_shape = (len(initial_pos), CONTROL_HORIZON, 2)

    # Handle controllable mask
    if controllable_mask is None:
        controllable_mask = np.ones(n_robots, dtype=bool)
    else:
        controllable_mask = controllable_mask.astype(bool)

    # Pad inputs to fixed size (6 robots) in numpy
    padded_max_speeds = np.full(6, 1e6)
    padded_max_speeds[:n_robots] = max_speeds
    
    # Apply controllable mask - set max_speeds to 0 for non-controllable robots
    effective_max_speeds = max_speeds.copy()
    effective_max_speeds[~controllable_mask] = 0.0
    padded_max_speeds[:n_robots] = effective_max_speeds

    if last_control_sequences is None:
        padded_last_control = np.zeros((6, CONTROL_HORIZON, 2))
    else:
        padded_last_control = np.zeros((6, CONTROL_HORIZON, 2))
        effective_last_control = last_control_sequences.copy()
        # Zero out controls for non-controllable robots
        effective_last_control[~controllable_mask] = 0.0
        padded_last_control[:n_robots] = effective_last_control

    field_bounds_obj = FieldBounds(
        field_length=float(field_geometry[0]),
        field_width=float(field_geometry[1]),
        penalty_area_depth=float(field_geometry[2]),
        penalty_area_width=float(field_geometry[3]),
    )

    r = _jitted(
        w=World(
            field_bounds_obj,
            EntityBatch(jnp.asarray(obstacles)),
            EntityBatch(jnp.asarray(initial_pos), jnp.asarray(initial_vel)),
            Entity(jnp.asarray(ball_pos)),
        ),
        targets=EntityBatch(jnp.asarray(target_pos)),
        max_speeds=jnp.asarray(padded_max_speeds),
        last_control_sequences=jnp.asarray(padded_last_control),
    )

    # Unpad the result in numpy
    r = Result(
        u=np.array(r.u)[:n_robots],
        traj=np.array(r.traj)[:n_robots],
        cost=float(r.cost),
        idx_by_cost=int(r.idx_by_cost),
    )

    return r


def solve_mpc_tbwrap(*args):
    try:
        result = solve_mpc(*args)
        return result.u, result.traj
    except Exception:
        repr = ""
        for arg in args:
            repr += f"{arg}\n\n---------------\n"

        shapes_repr = ""
        for a in args:
            if isinstance(a, float) or isinstance(a, int):
                shapes_repr += "() "
            elif a is None:
                shapes_repr += "None "
            else:
                shapes_repr += f"{a.shape} "
        raise RuntimeError(
            f"Traceback: {tb.format_exc(20)} with input: {repr} and shapes: {shapes_repr}"
        ) from None
