use nalgebra::Vector2;

// TODO: Fine tune the constants on real robot

/// The number of points in the path
const N_POINTS: usize = 5;

/// The spacing between points in the path in mm
const POINT_SPACING: f32 = 100.0;

/// The attractive force constant
const ALPHA: f32 = 0.00001;

/// The repulsive force constant
const BETA: f32 = 0.01;

/// The influence factor constants
const INFLUENCE_FACTOR: (f32, f32) = (5.0, 1.0);

/// A circular obstacle
#[derive(Debug, Clone)]
pub struct Obstacle {
    pub position: Vector2<f32>,
    pub radius: f32,
    pub velocity: Vector2<f32>,
}

impl Obstacle {
    /// Creates a new obstacle
    pub fn new(position: Vector2<f32>, radius: f32, velocity: Vector2<f32>) -> Self {
        Self {
            position,
            radius,
            velocity,
        }
    }

    /// Creates a new obstacle with 0 velocity
    pub fn new_without_vel(position: Vector2<f32>, radius: f32) -> Self {
        Self {
            position,
            radius,
            velocity: Vector2::new(0.0, 0.0),
        }
    }
}

/// Computes a path from `start` to `goal` avoiding `obstacles`.
///
/// Returns a computed trajectory of [`N_POINTS`] points, spaced by [`POINT_SPACING`].
///
/// This implementation uses [potential fields](https://medium.com/nerd-for-tech/local-path-planning-using-virtual-potential-field-in-python-ec0998f490af)
/// to compute the path. It does not take into account the robot's dimensions.
///
/// _TODO:_ Fine tune the constants
///
/// _TODO:_ Explore other path planning algorithms
pub fn compute_path(
    start: &Vector2<f32>,
    goal: &Vector2<f32>,
    obstacles: &[Obstacle],
    start_speed: Option<Vector2<f32>>,
) -> [Vector2<f32>; N_POINTS] {
    let mut path: [Vector2<f32>; N_POINTS] = [Vector2::new(0.0, 0.0); N_POINTS];
    let mut pos = start.clone();
    let start_speed = start_speed.unwrap_or(Vector2::new(0.0, 0.0));

    for index in 0..N_POINTS {
        let mut f: Vector2<f32> = (goal - pos)
            .try_normalize(f32::EPSILON)
            .unwrap_or(Vector2::zeros());
        let dist = (goal - pos).norm();
        let attractive_force = ALPHA * dist;
        f *= attractive_force;

        let (base_factor, speed_factor) = INFLUENCE_FACTOR;
        for obs in obstacles {
            let d = (obs.position - pos).norm();
            let relative_speed = (pos - obs.position)
                .try_normalize(f32::EPSILON)
                .unwrap_or(Vector2::zeros())
                .dot(&(start_speed - obs.velocity))
                .abs();
            let influence_radius =
                base_factor as f32 * obs.radius + relative_speed * speed_factor as f32;
            if d < influence_radius {
                let repulsive_force = (1.0 / ((d - 2.0 * obs.radius) + f32::EPSILON)
                    - 1.0 / ((influence_radius - 2.0 * obs.radius) + f32::EPSILON))
                    * (relative_speed * (BETA + 1.0));

                f += (pos - obs.position) * repulsive_force;
            }
        }

        pos += f.try_normalize(f32::EPSILON).unwrap_or(Vector2::zeros()) * POINT_SPACING;

        if index == 0 || (goal - pos).norm() < (goal - path[index - 1]).norm() {
            path[index] = pos.clone();
        } else {
            path[index] = path[index - 1];
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_obstacles() {
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(9000.0, 6000.0);
        let direction = (goal - start).normalize();
        let expected_points: [Vector2<f32>; 5] = [
            start + direction * 100.0,
            start + direction * 200.0,
            start + direction * 300.0,
            start + direction * 400.0,
            start + direction * 500.0,
        ];

        let path = compute_path(&start, &goal, &[], None);
        for i in 0..N_POINTS {
            assert!((path[i] - expected_points[i]).norm() <= 1e-4);
        }
    }

    #[test]
    fn test_1_obstacle_middle() {
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(3000.0, 5000.0);
        let obstacles = [Obstacle::new(
            Vector2::new(1500.0, 2500.0),
            80.0,
            Vector2::new(1000.0, 0.0),
        )];

        let path = compute_path(&start, &goal, &obstacles, None);

        let mut last_distance_to_goal = (goal - Vector2::new(0.0, 0.0)).norm();

        for point in &path {
            let distance_to_goal = (goal - *point).norm();
            assert!(distance_to_goal <= last_distance_to_goal + 1e-4);
            last_distance_to_goal = distance_to_goal;
            assert!((point - obstacles[0].position).norm() > obstacles[0].radius);
        }
    }

    #[test]
    fn test_2_obstacles_close_to_start() {
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(30000.0, 50000.0);
        let obstacles = [
            Obstacle::new_without_vel(Vector2::new(500.0, 0.0), 250.0),
            Obstacle::new_without_vel(Vector2::new(0.0, 700.0), 250.0),
        ];

        let path = compute_path(&start, &goal, &obstacles, None);

        let mut last_distance_to_goal = (goal - Vector2::new(0.0, 0.0)).norm();
        for point in &path {
            let distance_to_goal = (goal - *point).norm();
            assert!(distance_to_goal <= last_distance_to_goal + 1e-4);
            last_distance_to_goal = distance_to_goal;
            for obs in &obstacles {
                assert!((point - obs.position).norm() > obs.radius);
            }
        }
    }
}
