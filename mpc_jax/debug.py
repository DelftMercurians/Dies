#!/usr/bin/env python3

from typing import Literal
import jax
import jax.numpy as jnp
import jax.random as jr
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

from common import (
    ROBOT_RADIUS,
    trajectories_from_control,
    World,
    FieldBounds,
    Entity,
    EntityBatch,
    Control,
)


def visualize_mpc_debug(
    w: World,
    target_pos: np.ndarray,
    max_vel: float,
    resolution: int = 50,
    log_scale: bool = False,
):
    # Create a grid for visualization
    min_x, max_x, min_y, max_y = w.field_bounds.bounding_box()
    x_range = np.linspace(min_x, max_x, resolution)
    y_range = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x_range, y_range)

    # Create position grid as a flat array of [x, y] positions
    positions = np.stack([X.flatten(), Y.flatten()], axis=-1)

    # Vectorized cost function that evaluates at multiple positions
    @jax.jit
    def position_cost_vmap(positions, target, obstacles, field_bounds):
        dist_costs = jax.vmap(lambda p: distance_cost(p, target, 0))(positions)
        coll_costs = jax.vmap(lambda p: collision_cost(p, obstacles))(positions)
        bound_costs = jax.vmap(lambda p: boundary_cost(p, field_bounds))(positions)

        return dist_costs + coll_costs + bound_costs

    # Compute costs for all grid points in parallel
    costs_flat = np.array(
        position_cost_vmap(positions, target_pos, w.obstacles.position, w.field_bounds)
    )
    costs = costs_flat.reshape(resolution, resolution)

    # Get MPC control sequence and convert to trajectory
    controls = solve_mpc(
        w.robots.position,
        w.robots.velocity,
        target_pos,
        w.obstacles.position,
        w.field_bounds.bounding_box(),
        max_vel,
    )
    trajectories = trajectories_from_control(w, controls)

    # Extract positions from trajectories (skip time column)
    num_robots = trajectories.shape[0]
    robot_trajectories = [trajectories[i, :, 1:] for i in range(num_robots)]

    aspect_ratio = (max_x - min_x) / (max_y - min_y)
    fig_width = 16
    fig_height = fig_width / aspect_ratio
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_data = np.log1p(costs) if log_scale else costs
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

    # Plot start positions for all robots
    colors = ["r", "b"]
    for i in range(num_robots):
        color = colors[i % len(colors)]
        ax.plot(
            w.robots.position[i, 0],
            w.robots.position[i, 1],
            f"{color}s",
            markersize=14,
            label=f"Start {i + 1}",
        )

    # Plot target positions for all robots
    for i in range(min(target_pos.shape[0], num_robots)):
        color = colors[i % len(colors)]
        ax.plot(
            target_pos[i, 0],
            target_pos[i, 1],
            f"{color}*",
            markersize=20,
            label=f"Target {i + 1}",
        )

    # Plot trajectories for all robots
    for i, trajectory in enumerate(robot_trajectories):
        color = colors[i % len(colors)]
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            f"{color}-",
            linewidth=2,
            label=f"MPC Path {i + 1}",
        )

    ax.plot([], [], "o", color="cyan", markersize=6, label="0.5s markers")

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


def animate_moving_obstacle(initial_pos, target_pos):
    """Create an animation of the MPC debug plot with moving second obstacle."""
    # Animation parameters
    num_frames = 10

    # Fixed setup - two robots
    max_vel = 2000.0

    # First obstacle is stationary
    obstacle1 = np.array([500.0, 250.0])

    # Second obstacle moves around
    y_start, y_end = 900.0, 400.0

    all_obstacles = []
    all_trajectories = []
    for frame in tqdm(range(num_frames), desc="Computing frames"):
        # Calculate second obstacle position
        t = frame / (num_frames - 1)  # 0 to 1
        y_pos = y_start + (y_end - y_start) * t
        obstacle2 = np.array([400.0, y_pos])
        obstacles = np.array([obstacle1, obstacle2])
        all_obstacles.append(obstacles)

        w = World(FieldBounds(), EntityBatch(obstacles), EntityBatch(initial_pos))

        controls = solve_mpc(
            w.robots.position,
            w.robots.velocity,
            target_pos,
            w.obstacles.position,
            w.field_bounds.bounding_box(),
            max_vel,
        )
        trajectories = trajectories_from_control(w, controls)
        # Store trajectories for all robots
        robot_trajectories = []
        for robot_idx in range(trajectories.shape[0]):
            robot_trajectories.append(trajectories[robot_idx, :, 1:])
        all_trajectories.append(robot_trajectories)

    # Create static plot elements once
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pre-create circle patches for obstacles to reuse
    obstacle_circles = []
    for i in range(2):  # We have 2 obstacles
        circle = Circle((0, 0), ROBOT_RADIUS, color="black", alpha=0.5, fill=True)
        ax.add_patch(circle)
        obstacle_circles.append(circle)

    # Create line objects for trajectories to update more efficiently
    colors = ["m", "y"]
    start_colors = ["r", "b"]
    target_colors = ["r", "b"]

    trajectory_lines = []
    start_markers = []
    target_markers = []

    num_robots = initial_pos.shape[0]
    for i in range(num_robots):
        traj_color = colors[i % len(colors)]
        start_color = start_colors[i % len(start_colors)]
        target_color = target_colors[i % len(target_colors)]

        (trajectory_line,) = ax.plot(
            [], [], f"{traj_color}-", linewidth=2, label=f"MPC Path {i + 1}"
        )
        trajectory_lines.append(trajectory_line)

        (start_marker,) = ax.plot(
            initial_pos[i, 0],
            initial_pos[i, 1],
            f"{start_color}s",
            markersize=16,
            label=f"Start {i + 1}",
        )
        start_markers.append(start_marker)

        (target_marker,) = ax.plot(
            target_pos[i, 0],
            target_pos[i, 1],
            f"{target_color}*",
            markersize=20,
            label=f"Target {i + 1}",
        )
        target_markers.append(target_marker)

    # Set static properties once
    min_x, max_x, min_y, max_y = FieldBounds().bounding_box()
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

        # Update trajectories for all robots
        robot_trajectories = all_trajectories[frame]
        for i, trajectory in enumerate(robot_trajectories):
            trajectory_lines[i].set_data(trajectory[:, 0], trajectory[:, 1])

        # Update title
        ax.set_title(f"MPC with Moving Obstacle (Frame {frame + 1}/{num_frames})")

        return trajectory_lines + obstacle_circles + start_markers + target_markers

    anim = FuncAnimation(
        fig, animate, frames=num_frames, interval=200, repeat=True, blit=True
    )
    print("Saving the animation...")
    anim.save("mpc_moving_obstacle.gif", writer="pillow", fps=5)
    print("Animation saved to 'mpc_moving_obstacle.gif'")
    plt.close()


def plot_trajectories(
    w: World,
    targets: jax.Array,
    candidate_controls: jax.Array,
    optimal_trajectory,
    style: Literal["opt", "cand"] = "opt",
):
    """Plot all candidate trajectories with semi-transparent lines"""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Convert candidate trajectories to actual trajectories
    colors = ["m", "y"]
    for i, control_seq in enumerate(candidate_controls):
        trajectories_jax = trajectories_from_control(w, control_seq)

        # Plot trajectories for all robots
        num_robots = trajectories_jax.shape[0]
        for robot_idx in range(num_robots):
            trajectory = np.array(trajectories_jax)[
                robot_idx, :, 1:3
            ]  # Extract x, y positions
            color = colors[robot_idx % len(colors)]

            # Plot trajectory with semi-transparent line
            if style == "cand":
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    f"{color}-",
                    alpha=0.1,
                    linewidth=0.5,
                )
            else:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    f"{color}-",
                    alpha=0.2,
                    linewidth=0.8,
                )

    # Plot optimal trajectories for all robots
    num_robots = optimal_trajectory.shape[0]
    for robot_idx in range(num_robots):
        ax.plot(
            optimal_trajectory[robot_idx, :, 1],
            optimal_trajectory[robot_idx, :, 2],
            "k",
            linewidth=2,
        )

    # Plot obstacles as circles
    for obstacle in w.obstacles.position:
        circle = plt.Circle(obstacle, ROBOT_RADIUS, color="black", alpha=0.7)
        ax.add_patch(circle)

    # Plot start and target positions for all robots
    colors = ["r", "b"]
    target_colors = ["r", "b"]
    num_robots = w.robots.position.shape[0]

    for i in range(num_robots):
        color = colors[i % len(colors)]
        target_color = target_colors[i % len(target_colors)]

        ax.plot(
            w.robots.position[i, 0],
            w.robots.position[i, 1],
            f"{color}o",
            markersize=10,
            label=f"Start {i + 1}",
        )

        if i < targets.shape[0]:
            ax.plot(
                targets[i, 0],
                targets[i, 1],
                f"{target_color}*",
                markersize=15,
                label=f"Target {i + 1}",
            )

    # Set field bounds
    min_x, max_x, min_y, max_y = w.field_bounds.bounding_box()
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X position (mm)")
    ax.set_ylabel("Y position (mm)")
    ax.set_title("Trajectories")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"trajectories_{style}.png", dpi=150)
    print(f"Trajectories saved into 'trajectories_{style}.png'")


if __name__ == "__main__":
    # Set to True to use log scale for the cost plots
    USE_LOG_SCALE = False

    # Run visualization with two robots test case
    initial_pos = np.array([[-400.0, 550.0], [-300.0, 200.0]])
    target_pos = np.array([[1000.0, 400.0], [800.0, 700.0]])
    obstacles = np.array([[500.0, 250.0], [400, 600]])
    max_vel = 2000.0
    w = World(
        field_bounds=FieldBounds(),
        obstacles=EntityBatch(obstacles),
        robots=EntityBatch(initial_pos),
    )

    # Solve MPC
    optimal_control, candidate_controls, optimized_controls = solve_mpc(
        initial_pos,
        np.zeros((2, 2)),  # Two robots, so velocity shape is (2, 2)
        target_pos,
        obstacles,
        FieldBounds().bounding_box(),
        max_vel,
        with_aux=True,
    )
    optimal_trajectories = trajectories_from_control(w, optimal_control)

    # Plotting candidate trajectories
    plot_trajectories(
        w,
        target_pos,
        candidate_controls,
        optimal_trajectories,
        style="cand",
    )
    plot_trajectories(
        w,
        target_pos,
        optimized_controls,
        optimal_trajectories,
        style="opt",
    )

    # Generate visualizations with proper aspect ratio and optional log scale
    print(f"Generating visualizations with log_scale={USE_LOG_SCALE}")
    visualize_mpc_debug(
        w,
        target_pos,
        max_vel,
        log_scale=USE_LOG_SCALE,
    )

    # Plot the collision cost function separately
    plot_collision_cost(resolution=50, log_scale=USE_LOG_SCALE)

    # Create animation of moving obstacle
    animate_moving_obstacle(initial_pos, target_pos)

    print("JAX MPC test completed successfully!")
