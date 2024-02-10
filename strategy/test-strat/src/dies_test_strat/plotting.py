import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_static_obstacles(obstacles, pos_constraints, x_init, x_target):
    # Plot static obstacles
    for idx, obstacle in enumerate(obstacles):
        circle = plt.Circle((obstacle[0], obstacle[1]), obstacle[2], color='red', fill=True, label= f'Obstacle {idx}')
        plt.gca().add_patch(circle)

    # Plot start and end goal
    circle_init = plt.Circle((x_init[0], x_init[1]), 1, color='green', fill=True, label= f'Start Position')
    plt.gca().add_patch(circle_init)    
    circle_target = plt.Circle((x_target[0], x_target[1]), 1, color='green', fill=True, label= f'End Position')
    plt.gca().add_patch(circle_target)

    # Env Boundaries
    x = pos_constraints[0]
    y = pos_constraints[2]
    width = pos_constraints[1]-x
    height = pos_constraints[3]-x
    rectangle = patches.Rectangle((x, y), width, height, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rectangle)
    
    # plt properties
    plt.title('Static Simulation Environment')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(pos_constraints[0], pos_constraints[1])
    plt.ylim(pos_constraints[2], pos_constraints[3])
    plt.axis('equal')
    plt.grid()
    plt.show()