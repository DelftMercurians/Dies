import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, value_and_grad
from jaxtyping import PRNGKeyArray, Float, Array
import numpy as np
import optax
import os
import equinox as eqx
from tqdm import tqdm
import functools as ft
import matplotlib.pyplot as plt
from common import (
    ROBOT_RADIUS,
    LEARNING_RATE,
    CONTROL_HORIZON,
    MAX_ITERATIONS,
    UPDATE_CLIP,
    N_CANDIDATE_TRAJECTORIES,
    trajectories_from_control,
    World,
    Control,
    Entity,
    EntityBatch,
    FieldBounds,
)

from costs import (
    distance_cost,
    collision_cost,
    boundary_cost,
    velocity_constraint_cost,
    control_effort_cost,
)


def mpc_cost_function(
    u: Control,
    w: World,
    targets: EntityBatch,
    max_speed: float,
):
    # Generate trajectory from control sequence
    trajectories = trajectories_from_control(w, u)
    n_robots = len(w.robots)

    # Compute position-based costs over the trajectory
    def position_cost_fn(raw_traj, target):
        t, pos_x, pos_y, vel_x, vel_y = raw_traj
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        d_cost = distance_cost(robot.position, target.after(t).position, 5)
        c_cost = collision_cost(robot.position, w.obstacles.after(t).position)
        b_cost = boundary_cost(robot.position, w.field_bounds)
        vc_cost = velocity_constraint_cost(robot.velocity, max_speed)
        return d_cost + c_cost + b_cost + vc_cost

    def collective_position_cost_fn(traj_slice, idx):
        t, pos_x, pos_y, vel_x, vel_y = traj_slice[idx]
        robot = Entity(jnp.array([pos_x, pos_y]), jnp.array([vel_x, vel_y]))
        mask = jnp.ones((n_robots,)).at[idx].set(0)
        ours_c_cost = (
            collision_cost(
                robot.position,
                jax.lax.stop_gradient(traj_slice[:, 1:3]),
                mask=mask,
            )
            * 0.2
        )
        return ours_c_cost

    # Skip initial position (i=0) and compute costs for trajectory steps
    total_position_cost = jax.vmap(
        lambda traj, target: jax.vmap(eqx.Partial(position_cost_fn, target=target))(
            traj
        )
    )(trajectories[:, 1:], targets).sum()

    total_collective_cost = jax.vmap(
        lambda traj_slice: jax.vmap(
            eqx.Partial(collective_position_cost_fn, traj_slice)
        )(jnp.arange(n_robots))
    )(trajectories.reshape((-1, 2, 5))).sum()

    return total_position_cost + total_collective_cost


def clip_vel(vel: jnp.ndarray, limit: float | Float[Array, ""]):
    speed = jnp.sqrt((vel**2).sum() + 1e-9)
    return jnp.where(speed > limit, vel * (limit / speed), vel)


def generate_candidate_control(
    key: PRNGKeyArray,
    initial_pos: Float[Array, "2"],
    target_pos: Float[Array, "2"],
    max_speed: float,
) -> Float[Array, f"{CONTROL_HORIZON} 2"]:
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    # Generate pseudo-target with normal noise (std 50cm = 500mm)
    noise = jr.normal(k1, (2,)) * 500.0
    pseudo_target = target_pos + noise

    # Stage 1: Go to 75%ish of distance in 1 second (5 steps since DT=0.2)
    ratio = jr.uniform(k6, (), minval=0.4, maxval=1.2)
    intermediate_target = initial_pos + ratio * (pseudo_target - initial_pos)

    # Calculate required velocity to reach 75% in a few seconds
    few_seconds = jr.uniform(k5, (), minval=2.0, maxval=5.0)
    required_vel_stage1 = (intermediate_target - initial_pos) / few_seconds
    required_vel_stage1 = clip_vel(required_vel_stage1, max_speed)

    # Add noise to stage 1 velocity
    noise1 = jr.normal(k2, (2,)) * 200.0
    first_stage_random_vel_clip_factor = jr.uniform(k3, (), minval=0.2, maxval=1.0)
    stage1_vel = clip_vel(
        required_vel_stage1 + noise1, max_speed * first_stage_random_vel_clip_factor
    )

    # Stage 2: Go from intermediate to full pseudo-target
    remaining_distance = jnp.linalg.norm(pseudo_target - intermediate_target)
    remaining_time = jnp.maximum(few_seconds, remaining_distance / max_speed)
    required_vel_stage2 = (pseudo_target - intermediate_target) / remaining_time
    required_vel_stage2 = clip_vel(required_vel_stage2, max_speed)

    # Add noise to stage 2 velocity
    noise2 = jr.normal(k4, (2,)) * 200.0
    stage2_vel = clip_vel(required_vel_stage2 + noise2, max_speed)

    # Combine the two thingies
    stage1_steps = CONTROL_HORIZON // 2
    stage2_steps = CONTROL_HORIZON // 2

    control_sequence = jnp.concatenate(
        [
            jnp.tile(stage1_vel[None, :], (stage1_steps, 1)),
            jnp.tile(stage2_vel[None, :], (stage2_steps, 1)),
        ]
    )

    return control_sequence


def solve_mpc_jax(
    w: World,
    targets: EntityBatch,
    max_speed: float,
    max_iterations: int,
    learning_rate: float,
    n_candidates: int,
    key: PRNGKeyArray | None = None,
) -> tuple:
    n_robots = len(w.robots)

    if key is None:
        key = jr.PRNGKey(0)

    # Generate candidate trajectories
    keys = jr.split(key, n_candidates)
    candidate_controls = jax.vmap(
        lambda k: jax.vmap(
            eqx.Partial(generate_candidate_control, max_speed=max_speed)
        )(
            jr.split(k, n_robots),
            w.robots.position,
            targets.after(3).position,  # this is a heuristic, since why not
        )
    )(keys)
    candidate_controls = candidate_controls.reshape(
        (n_candidates, n_robots, CONTROL_HORIZON, 2)
    )

    # Optimize each candidate trajectory via (full-batch) gradient descent
    def optimize_control(u):
        # Initialize optimizer for this trajectory
        schedule = optax.linear_schedule(
            learning_rate, learning_rate / 10.0, max_iterations
        )
        optimizer = optax.chain(
            optax.sgd(schedule, momentum=0.5, nesterov=True),
            optax.clip_by_global_norm(UPDATE_CLIP),
        )
        opt_state = optimizer.init(u)

        def optimization_step(carry, _):
            u, opt_state = carry

            # Compute gradient wrt complete cost function
            cost, grad_val = eqx.filter_value_and_grad(mpc_cost_function)(
                u, w, targets, max_speed
            )

            # Do an optimization step
            updates, opt_state = optimizer.update(grad_val, opt_state)
            u = optax.apply_updates(u, updates)

            # Project the control back to the "valid" domain
            u = jax.vmap(lambda vel: clip_vel(vel, max_speed))(u)

            return (u, opt_state), cost

        (u, _), costs = jax.lax.scan(
            optimization_step,
            (u, opt_state),
            None,
            length=max_iterations,
        )

        return u, costs[-1]  # Return final control and final cost

    # Optimize all candidate controls in parallel
    optimized_controls, final_costs = jax.vmap(optimize_control)(candidate_controls)

    # Select the best trajectory (lowest cost)
    best_idx = jnp.argmin(final_costs)
    best_control = optimized_controls[best_idx]

    return best_control, candidate_controls, optimized_controls


solve_mpc_jitted = eqx.filter_jit(solve_mpc_jax)


def solve_mpc(
    initial_pos: np.ndarray,
    initial_vel: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    field_bounds: np.ndarray | None,
    max_vel: float,
    max_iterations: int = MAX_ITERATIONS,
    learning_rate: float = LEARNING_RATE,
    n_candidates: int = N_CANDIDATE_TRAJECTORIES,
    key: PRNGKeyArray | None = None,
    with_aux: bool = False,
) -> tuple | np.ndarray:
    out = solve_mpc_jitted(
        World(
            FieldBounds(),
            EntityBatch(jnp.asarray(obstacles)),
            EntityBatch(jnp.asarray(initial_pos), jnp.asarray(initial_vel)),
            Entity(jnp.zeros((2,))),
        ),
        EntityBatch(jnp.asarray(target_pos)),
        max_vel,
        int(max_iterations),
        float(learning_rate),
        int(n_candidates),
        key,
    )

    if with_aux:
        return out
    else:
        return out[0]


def test_simple_case():
    # Simple test case
    initial_pos = np.array([0.0, 0.0])
    initial_vel = np.array([0.0, 0.0])
    target_pos = np.array([1000.0, 500.0])  # 1m forward, 0.5m right
    obstacles = np.array([[500.0, 250.0]])  # One obstacle in the way
    field_bounds = np.array([-2000.0, 2000.0, -1000.0, 1000.0])  # 4m x 2m field
    max_vel = 4000.0  # 2 m/s

    # Solve MPC
    optimal_control = solve_mpc(
        initial_pos, initial_vel, target_pos, obstacles, field_bounds, max_vel
    )

    print(f"Control: {np.linalg.norm(optimal_control):.2f}")
    return optimal_control[0]


if __name__ == "__main__":
    test_simple_case()
    import time

    t = time.time()
    test_simple_case()
    print(f"Took {(time.time() - t) * 1000:.1f}ms")
