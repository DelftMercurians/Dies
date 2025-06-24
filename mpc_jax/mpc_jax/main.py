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
    calculate_mismatch_score,
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
    last_traj,
):
    k1, k2, k3 = jr.split(key, 3)
    w = w.noisy(k1)
    cfg = cfg.noisy(k2)
    u = add_control_noise(k3, u)
    trajectories = trajectories_from_control(w, u, cfg.delay)
    n_robots = len(w.robots)

    # Compute position-based costs over the trajectory
    def position_cost_fn(raw_traj, max_speed, target):
        t, pos_x, pos_y, vel_x, vel_y = raw_traj
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        # Account for delay in target position prediction
        delayed_target_pos = target.after(t + cfg.delay).position
        d_cost = distance_cost(robot.position, delayed_target_pos, t)
        # collision with obstacles (non-controllable robots) costs
        c_cost = 0
        if len(w.obstacles) != 0:
            # Account for delay in obstacle position prediction
            delayed_obstacle_pos = w.obstacles.after(t + cfg.delay).position
            c_cost = collision_cost(
                robot.position,
                delayed_obstacle_pos,
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
        effort_cost = jnp.sqrt((u**2).sum()) * 1e-6
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
        )

    def collective_position_cost_fn(traj_slice, idx):
        # collisions with our own robots cost
        t, pos_x, pos_y, vel_x, vel_y = traj_slice[idx]
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        mask = jnp.ones((n_robots,)).at[idx].set(0)
        obstacles = jax.lax.stop_gradient(traj_slice[:, 1:3])
        ours_c_cost = collision_cost(
            robot.position,
            obstacles,
            mask=mask,
            min_safe_distance=cfg.obstacle_min_safe_distance,
            no_cost_distance=cfg.obstacle_no_cost_distance,
        )
        return ours_c_cost * 2

    # Skip initial position (i=0) and compute costs for trajectory steps
    total_position_cost = jax.vmap(
        lambda traj, target, max_speed: jax.vmap(
            eqx.Partial(position_cost_fn, max_speed=max_speed, target=target)
        )(traj)
    )(trajectories, targets, max_speeds)

    total_collective_cost = jax.vmap(
        lambda traj_slice: jax.vmap(
            eqx.Partial(collective_position_cost_fn, traj_slice)
        )(jnp.arange(n_robots)),
        in_axes=1,
    )(trajectories).reshape(n_robots, -1)

    tie_breaker_cost = 0
    """
    if last_traj is not None and last_traj.shape == trajectories.shape:
        # Match nearest time points and compute (x,y) position differences
        # trajectories shape: (n_robots, n_time_steps, 5) where 5 = (time, x, y, vx, vy)
        def compute_trajectory_diff(current_traj, last_traj):
            # Extract time and position for both trajectories
            current_times = current_traj[:, 0]  # (n_time_steps,)
            current_pos = current_traj[:, 1:3]  # (n_time_steps, 2) for (x, y)
            last_times = last_traj[:, 0]  # (n_time_steps,)
            last_pos = last_traj[:, 1:3]  # (n_time_steps, 2) for (x, y)

            # For each point in current trajectory, find nearest time in last trajectory
            def find_nearest_match(curr_time, curr_pos):
                time_diffs = jnp.abs(last_times - curr_time)
                nearest_idx = jnp.argmin(time_diffs)
                matched_pos = last_pos[nearest_idx]
                pos_diff = curr_pos - matched_pos
                return jnp.sqrt(pos_diff**2 + 1e-8)

            # Compute position differences for all time points
            diffs = jax.vmap(find_nearest_match)(current_times, current_pos)
            return diffs.mean()

        # Compute differences for all robots
        robot_diffs = jax.vmap(compute_trajectory_diff)(trajectories, last_traj)
        diff = robot_diffs.mean()
        tie_breaker_cost = diff * 1e-3
    """

    return (total_position_cost + total_collective_cost + tie_breaker_cost).sum()


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
    max_iterations: int,
    learning_rate: float,
    n_candidates: int,
    key: PRNGKeyArray | None = None,
    last_traj: jax.Array | None = None,
    last_control_sequences: jax.Array | None = None,
    cfg: MPCConfig | None = None,
) -> Result:
    n_robots = len(w.robots)

    if key is None:
        key = jr.PRNGKey(0)

    if cfg is None:
        cfg = MPCConfig()

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

    if continuity_candidates > 0:
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
    else:
        candidate_controls = random_candidate_controls

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
                last_traj=last_traj,
            )
        )(jr.split(key, BATCH_SIZE)).mean()

    # Optimize each candidate trajectory via (full-batch) gradient descent
    def optimize_control(u, key, cfg: MPCConfig = MPCConfig()):
        # Initialize optimizer for this trajectory
        lr_schedule = optax.linear_schedule(
            learning_rate, learning_rate / 8.0, max_iterations
        )
        optimizer = optax.chain(
            optax.adabelief(learning_rate=lr_schedule, b1=0.8, b2=0.8),
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
            stochastic_key, optimized_controls, w, targets, cfg
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
        candidate_controls=candidate_controls,
        optimized_controls=optimized_controls,
        cost=best_cost,
        idx_by_cost=best_idx,
    )


def pprint(arr):
    out = "["
    for v in arr.ravel():
        out += f"{float(v):.2f} "
    return out[:-1] + "]" if len(out) != 1 else "[]"


def solve_mpc(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    ball_pos: np.ndarray | None,
    field_bounds: np.ndarray | None,
    max_speed: np.ndarray,
    last_control_sequences: np.ndarray | None = None,
    last_traj: np.ndarray | None = None,
    dt: np.ndarray | None = np.array(0.02),
    max_iterations: int = MAX_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    n_candidates: int = N_CANDIDATE_TRAJECTORIES,
    key: PRNGKeyArray | None = None,
) -> Result:
    n_robots = len(initial_pos)
    assert len(initial_vel) == n_robots, f"{len(initial_vel)} != {n_robots}"
    assert len(target_pos) == n_robots, f"{len(target_pos)} != {n_robots}"
    assert len(max_speed) == n_robots, f"{len(max_speed)} != {n_robots}"

    last_controls_jax = (
        None if last_control_sequences is None else jnp.asarray(last_control_sequences)
    )
    ctrl_shape = (len(initial_pos), CONTROL_HORIZON, 2)
    if (
        last_control_sequences is not None
        and last_control_sequences.shape != ctrl_shape
    ):
        warnings.warn(
            f"Last control sequence had shape {last_control_sequences.shape}, but was expected to have shape {ctrl_shape}. Disabling continuity."
        )

    # Handle ball position - use provided ball_pos or default to far away
    ball_position = jnp.array([1e6, 1e6]) if ball_pos is None else jnp.asarray(ball_pos)
    if last_traj is not None:
        last_traj = np.array(last_traj, dtype=np.float32)
        last_traj[:, :, 0] = last_traj[:, :, 0] - dt

    r = eqx.filter_jit(solve_mpc_jax)(
        w=World(
            FieldBounds(),
            EntityBatch(jnp.asarray(obstacles)),
            EntityBatch(jnp.asarray(initial_pos), jnp.asarray(initial_vel)),
            Entity(ball_position),
        ),
        targets=EntityBatch(jnp.asarray(target_pos)),
        max_speeds=jnp.asarray(max_speed),
        max_iterations=int(max_iterations),
        learning_rate=float(learning_rate),
        n_candidates=int(n_candidates),
        key=key,
        last_control_sequences=None
        if last_controls_jax is None
        else jnp.asarray(last_controls_jax),
        last_traj=None if last_traj is None else jnp.asarray(last_traj),
        cfg=MPCConfig(),  # Use default config with 0.02s delay
    )

    # Calculate mismatch score
    # mismatch_score = calculate_mismatch_score(
    #    r.traj, jnp.asarray(initial_pos), dt if dt is not None else None
    # )

    # print(f"Lower is better: {r.idx_by_cost} / {N_CANDIDATE_TRAJECTORIES}")
    # print(f"Mismatch score: \t{mismatch_score:.0f}mm")

    if np.isinf(r.cost) or np.isnan(r.cost):
        warnings.warn(
            "Cost if infinite (or nan), which means there is no collision-free resolution found by MPC. Proceeding with the best available trajectory."
        )
    return r


def solve_mpc_tbwrap(*args):
    try:
        result = solve_mpc(*args)
        return result.u, result.traj
    except Exception:
        raise RuntimeError(
            f"Traceback: {tb.format_exc(20)} with input: {args}"
        ) from None
