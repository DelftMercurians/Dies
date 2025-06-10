#!/usr/bin/env python3

import jax
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import numpy as np
import optax
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from tqdm import tqdm
import functools as ft

from main import (
    distance_cost,
    collision_cost,
    boundary_cost,
    mpc_cost_function,
    solve_mpc,
)

from common import DT, ROBOT_RADIUS, trajectory_from_control


def visualize_mpc_debug(
    initial_pos: np.ndarray,
    target_pos: np.ndarray,
    obstacles: np.ndarray,
    field_bounds: np.ndarray,
    max_vel: float,
    resolution: int = 50,
    log_scale: bool = False,
):
    # Create a grid for visualization
    min_x, max_x, min_y, max_y = field_bounds
    x_range = np.linspace(min_x, max_x, resolution)
    y_range = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Create position grid as a flat array of [x, y] positions
    positions = np.stack([X.flatten(), Y.flatten()], axis=-1)

    # Initial velocity (zero for this visualization)
    initial_vel = np.zeros(2)

    # Vectorized cost function that evaluates at multiple positions
    @jax.jit
    def position_cost_vmap(positions, target, obstacles, field_bounds):
        dist_costs = jax.vmap(lambda p: distance_cost(p, target, 0))(positions)
        coll_costs = jax.vmap(lambda p: collision_cost(p, obstacles))(positions)
        bound_costs = jax.vmap(lambda p: boundary_cost(p, field_bounds))(positions)

        return dist_costs + coll_costs + bound_costs

    # Compute costs for all grid points in parallel
    print("Computing costs with jitted vmap...")
    positions_jax = jnp.array(positions)
    target_jax = jnp.array(target_pos)
    obstacles_jax = jnp.array(obstacles)
    field_bounds_jax = jnp.array(field_bounds)

    costs_flat = np.array(
        position_cost_vmap(positions_jax, target_jax, obstacles_jax, field_bounds_jax)
    )
    costs = costs_flat.reshape(resolution, resolution)

    # Compute MPC trajectory
    trajectory_positions = [initial_pos]
    current_pos = initial_pos
    current_vel = initial_vel

    # Get MPC control sequence and convert to trajectory
    all_controls = solve_mpc(
        current_pos, current_vel, target_pos, obstacles, field_bounds, max_vel
    )
    # Use trajectory_from_control to get trajectory from control sequence
    trajectory_jax = trajectory_from_control(
        jnp.array(all_controls), jnp.array(current_pos)
    )
    trajectory_with_time = np.array(trajectory_jax)

    # Extract positions from trajectory (skip time column)
    trajectory = trajectory_with_time[:, 1:]  # Shape: (horizon+1, 2)

    # Create the figure with aspect ratio matching the field dimensions
    aspect_ratio = (max_x - min_x) / (max_y - min_y)
    fig_width = 12
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Apply log scale to costs if requested
    plot_data = np.log1p(costs) if log_scale else costs

    # Cost heatmap
    im = ax.imshow(
        plot_data,
        extent=[min_x, max_x, min_y, max_y],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )

    # Add colorbar with appropriate label
    colorbar_label = "Cost (log scale)" if log_scale else "Cost"
    fig.colorbar(im, ax=ax, label=colorbar_label)

    ax.plot(initial_pos[0], initial_pos[1], "bs", markersize=16, label="Start")
    ax.plot(target_pos[0], target_pos[1], "g*", markersize=20, label="Target")

    # Plot trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        "w-",
        linewidth=1,
        label="MPC Path",
    )

    # Add time markers every 0.5 seconds
    if len(trajectory) > 1:
        time_marker_interval = 0.5  # seconds
        marker_step = int(time_marker_interval / DT)  # convert to trajectory steps

        for i in range(0, len(trajectory), marker_step):
            if i < len(trajectory):
                ax.plot(
                    trajectory[i, 0],
                    trajectory[i, 1],
                    "o",
                    color="cyan",
                    markersize=6,
                )

    # Add legend entry for time markers
    ax.plot(
        [],
        [],
        "o",
        color="cyan",
        markersize=6,
        label="0.5s markers",
    )

    # Add legend and labels
    ax.set_title("MPC Cost Heatmap")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.legend()

    plt.tight_layout()
    plt.savefig("mpc_debug_visualization.png")
    print("Visualization saved to 'mpc_debug_visualization.png'")


def plot_collision_cost(
    resolution: int = 100,
    x_range: tuple = (-500, 500),
    y_range: tuple = (-500, 500),
    log_scale: bool = False,
) -> None:
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Note: The original had a transposition issue with indices [i,j] vs [j,i]
    # We'll fix that in the reshaping step
    positions = np.stack([X.T.flatten(), Y.T.flatten()], axis=-1)

    # Convert to JAX arrays
    obstacles = jnp.zeros([1, 2])
    positions_jax = jnp.array(positions)

    # Define and jit the vectorized collision cost function
    @jax.jit
    def collision_cost_vmap(positions, obstacles):
        return jax.vmap(lambda pos: collision_cost(pos, obstacles))(positions)

    # Vectorized collision cost computation
    print("Computing collision costs with jitted vmap...")
    costs_flat = np.array(collision_cost_vmap(positions_jax, obstacles))

    # Reshape to grid, preserving the original orientation
    costs = costs_flat.reshape(resolution, resolution)

    fig, ax = plt.subplots(figsize=(12, 12))

    # Apply log scale if requested
    plot_data = np.log1p(costs) if log_scale else costs

    # Plot heatmap
    im = ax.imshow(
        plot_data,
        extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
        origin="lower",
        cmap="viridis",
        aspect="auto",
    )

    # Add colorbar with appropriate label
    colorbar_label = "Collision Cost (log scale)" if log_scale else "Collision Cost"
    fig.colorbar(im, ax=ax, label=colorbar_label)

    # Add labels
    ax.set_title("Collision Cost Function Visualization")
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")

    plt.tight_layout()
    plt.savefig("collision_cost_visualization.png")
    print("Collision cost visualization saved to 'collision_cost_visualization.png'")


def animate_moving_obstacle():
    """Create an animation of the MPC debug plot with moving second obstacle."""
    # Animation parameters
    num_frames = 10

    # Fixed setup
    initial_pos = np.array([-300.0, 0.0])
    target_pos = np.array([700.0, 500.0])
    field_bounds = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    max_vel = 2000.0

    # First obstacle is stationary
    obstacle1 = np.array([500.0, 250.0])

    # Second obstacle moves between [400, 800] and [400, 400]
    y_start, y_end = 800.0, 400.0

    # Pre-compute all obstacle positions and trajectories for better performance
    print("Pre-computing trajectories for animation...")
    all_obstacles = []
    all_trajectories = []

    for frame in tqdm(range(num_frames), desc="Computing frames"):
        # Calculate second obstacle position
        t = frame / (num_frames - 1)  # 0 to 1
        y_pos = y_start + (y_end - y_start) * t
        obstacle2 = np.array([400.0, y_pos])
        obstacles = np.array([obstacle1, obstacle2])
        all_obstacles.append(obstacles)

        # Compute trajectory for this frame
        current_pos = initial_pos
        current_vel = np.zeros(2)

        all_controls = solve_mpc(
            current_pos, current_vel, target_pos, obstacles, field_bounds, max_vel
        )

        # Use trajectory_from_control to get trajectory from control sequence
        trajectory_jax = trajectory_from_control(
            jnp.array(all_controls), jnp.array(current_pos)
        )
        trajectory_with_time = np.array(trajectory_jax)
        # Extract positions from trajectory (skip time column)
        trajectory_positions = trajectory_with_time[:, 1:]  # Shape: (horizon+1, 2)
        all_trajectories.append(trajectory_positions)

    # Create static plot elements once
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pre-create circle patches for obstacles to reuse
    obstacle_circles = []
    for i in range(2):  # We have 2 obstacles
        circle = Circle((0, 0), ROBOT_RADIUS, color="black", alpha=0.5, fill=True)
        ax.add_patch(circle)
        obstacle_circles.append(circle)

    # Create line objects for trajectory to update more efficiently
    (trajectory_line,) = ax.plot([], [], "r-", linewidth=2, label="MPC Path")
    (start_marker,) = ax.plot(
        initial_pos[0], initial_pos[1], "bs", markersize=16, label="Start"
    )
    (target_marker,) = ax.plot(
        target_pos[0], target_pos[1], "g*", markersize=20, label="Target"
    )

    # Set static properties once
    min_x, max_x, min_y, max_y = field_bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.legend()

    def animate(frame):
        # Update obstacle positions
        obstacles = all_obstacles[frame]
        for i, obstacle in enumerate(obstacles):
            obstacle_circles[i].center = obstacle

        # Update trajectory
        trajectory = all_trajectories[frame]
        trajectory_line.set_data(trajectory[:, 0], trajectory[:, 1])

        # Update title
        ax.set_title(f"MPC with Moving Obstacle (Frame {frame + 1}/{num_frames})")

        return [trajectory_line] + obstacle_circles + [start_marker, target_marker]

    anim = FuncAnimation(
        fig, animate, frames=num_frames, interval=200, repeat=True, blit=True
    )
    print("Saving the animation...")
    anim.save("mpc_moving_obstacle.gif", writer="pillow", fps=5)
    print("Animation saved to 'mpc_moving_obstacle.gif'")
    plt.close()


if __name__ == "__main__":
    # Set to True to use log scale for the cost plots
    USE_LOG_SCALE = False

    # Run visualization with the simple test case
    initial_pos = np.array([-300.0, 0.0])
    target_pos = np.array([700.0, 500.0])
    obstacles = np.array([[500.0, 250.0], [400, 600]])
    field_bounds = np.array([-1000.0, 1000.0, -1000.0, 1000.0])
    max_vel = 2000.0

    # Generate visualizations with proper aspect ratio and optional log scale
    print(f"Generating visualizations with log_scale={USE_LOG_SCALE}")
    visualize_mpc_debug(
        initial_pos,
        target_pos,
        obstacles,
        field_bounds,
        max_vel,
        log_scale=USE_LOG_SCALE,
    )

    # Plot the collision cost function separately
    plot_collision_cost(resolution=50, log_scale=USE_LOG_SCALE)

    # Create animation of moving obstacle
    animate_moving_obstacle()

    print("JAX MPC test completed successfully!")
