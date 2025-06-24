import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PRNGKeyArray, Float, Array
import equinox as eqx

from .common import (
    BALL_RADIUS,
    ROBOT_RADIUS,
    TRAJECTORY_RESOLUTION,
    get_dt_schedule,
    single_trajectory_from_control,
    World,
    Control,
    Entity,
    EntityBatch,
    MPCConfig,
)


def add_control_noise(
    key: PRNGKeyArray,
    control: Control,
    noise_scale: float = 30.0,  # mm/s
) -> Control:
    """Add Gaussian noise to control sequence."""
    noise = jr.normal(key, control.shape) * noise_scale
    return control + noise


def point_vs_point_collision(a, b, min_safe_distance):
    """Check collision between two points."""
    assert a.shape == (2,), a.shape
    assert b.shape == (2,), b.shape
    return jnp.linalg.norm(a - b) < min_safe_distance


def traj_vs_entity_collision(traj, entity, min_safe_distance) -> jax.Array:
    """Check collision between trajectory and entity."""

    def check_point_collision(raw_value):
        t, pos_x, pos_y, vel_x, vel_y = raw_value
        obstacle_positions = entity.after(t).position
        if len(obstacle_positions.shape) == 1:
            obstacle_positions = obstacle_positions[None, :]
        collisions = jax.vmap(
            eqx.Partial(
                point_vs_point_collision,
                jnp.array([pos_x, pos_y]),
                min_safe_distance=min_safe_distance,
            )
        )(obstacle_positions)
        return collisions

    collisions = jax.vmap(check_point_collision)(traj)
    return collisions


def traj_vs_batch_collision(
    trajectory: jax.Array, obstacles: EntityBatch, min_safe_distance
) -> jax.Array:
    """Check collision between trajectory and batch of entities."""
    if len(obstacles) == 0:
        return jnp.array(False)

    return jax.vmap(lambda obj: traj_vs_entity_collision(trajectory, obj, min_safe_distance))(
        obstacles
    )


def multipoint_collision(points, min_safe_distance):
    """Check collision between multiple points at same time step."""
    diff = points[:, None] - points[None, :]
    dist = jnp.linalg.norm(diff, axis=-1)
    dist = dist + jnp.eye(len(points)) * 1e6  # no self-collisions
    return dist < min_safe_distance


def many_traj_collision(trajs, min_safe_distance):
    """Check collision between multiple trajectories."""
    collisions = jax.vmap(
        eqx.Partial(multipoint_collision, min_safe_distance=min_safe_distance), in_axes=1
    )(trajs[:, :, 1:3])
    return collisions


def compute_distance_integral(trajectory: jax.Array, target: Entity) -> jax.Array:
    """Compute integral of distance to target over trajectory."""
    times = trajectory[:, 0]
    positions = trajectory[:, 1:3]

    target_positions = jax.vmap(lambda t: target.after(t).position)(times)
    distances = jnp.linalg.norm(positions - target_positions, axis=1)

    # Trapezoidal rule integration
    dt_values = jnp.diff(times)
    avg_distances = 0.5 * (distances[:-1] + distances[1:])
    integral = jnp.sum(avg_distances * dt_values[:, None])

    return integral


def score_single_trajectory_sample(
    key: PRNGKeyArray,
    control: Control,
    world: World,
    targets: EntityBatch,
    cfg: MPCConfig,
) -> float:
    """Score a single noisy trajectory sample."""
    k1, k2, k3 = jr.split(key, 3)

    # Add noise to control
    noisy_control = add_control_noise(k1, control)

    # Create noisy config and world using the new methods
    noisy_cfg = cfg.noisy(k2)
    noisy_world = world.noisy(k3)
    time_discount = jnp.concatenate(
        [jnp.ones((TRAJECTORY_RESOLUTION + 1,)), jnp.exp(-get_dt_schedule(upscaled=True))]
    )

    # Generate trajectories with noisy parameters
    trajectories = jax.vmap(
        lambda ctrl, pos, vel: single_trajectory_from_control(ctrl, pos, vel, noisy_cfg.delay)
    )(noisy_control, noisy_world.robots.position, noisy_world.robots.velocity)

    # Check for collisions
    obstacle_collisions = (
        jax.vmap(
            lambda traj: traj_vs_batch_collision(
                traj, noisy_world.obstacles, min_safe_distance=2.2 * ROBOT_RADIUS
            )
        )(trajectories).astype(jnp.float32)
        * time_discount[None, None, :, None]
    )

    self_robot_collisions = (
        many_traj_collision(trajectories, min_safe_distance=2.2 * ROBOT_RADIUS).astype(jnp.float32)
        * time_discount[:, None, None]
    )

    ball_collisions = (
        jax.vmap(
            lambda traj: traj_vs_entity_collision(
                traj, noisy_world.ball, min_safe_distance=ROBOT_RADIUS * 1.2 + BALL_RADIUS
            )
        )(trajectories).astype(jnp.float32)
        * time_discount[None, :, None]
    )

    # Count total collisions
    total_collisions = (
        obstacle_collisions.sum() + self_robot_collisions.sum() + ball_collisions.sum()
    )

    collision_penalty = total_collisions * 1e12

    # Compute distance integral for collision-free samples
    robot_integrals = jax.vmap(lambda traj, target: compute_distance_integral(traj, target))(
        trajectories, targets
    )
    distance_score = jnp.sum(robot_integrals)

    return distance_score + collision_penalty


def stochastic_trajectory_scoring(
    key_or_keys,
    control: Control,
    world: World,
    targets: EntityBatch,
    cfg: MPCConfig,
    n_samples: int = 20,
    with_shared_noise: bool = False,
) -> float:
    if with_shared_noise:
        assert len(key_or_keys) == n_samples, (
            f"key_or_keys must be an array of keys when with_shared_noise=True of size {n_samples}, was {key_or_keys.shape}"
        )
        sample_keys = key_or_keys
        n_samples = len(sample_keys)
    else:
        sample_keys = jr.split(key_or_keys, n_samples)

    # Score all samples
    scores = jax.vmap(lambda k: score_single_trajectory_sample(k, control, world, targets, cfg))(
        sample_keys
    )

    # Sort scores and take trimmed mean (middle 16 out of 20)
    sorted_scores = jnp.sort(scores)
    n_trim = 2  # trim two values from each end
    trimmed_scores = sorted_scores[n_trim : n_samples - n_trim]

    return jnp.mean(trimmed_scores)


def select_best_stochastic_trajectory(
    key: PRNGKeyArray,
    optimized_controls: jax.Array,
    world: World,
    targets: EntityBatch,
    cfg: MPCConfig,
    n_samples: int = 50,
) -> tuple[int, float]:
    n_candidates = len(optimized_controls)
    sample_keys = jr.split(key, n_samples)

    # Score all candidates using the same noise sequences
    candidate_scores = jax.vmap(
        lambda control: stochastic_trajectory_scoring(
            sample_keys, control, world, targets, cfg, n_samples, with_shared_noise=True
        )
    )(optimized_controls)

    # Select candidate with minimum score
    best_idx = jnp.argmin(candidate_scores)
    best_score = candidate_scores[best_idx]

    return best_idx, best_score
