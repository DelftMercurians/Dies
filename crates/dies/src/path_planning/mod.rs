extern crate nalgebra as na;
use na::Vector2;
extern crate float_cmp;
use float_cmp::approx_eq;

/// The number of points in the path
const N_POINTS: usize = 5;

/// The spacing between points in the path in meters
const POINT_SPACING: f32 = 0.1;

/// A circular obstacle
pub struct Obstacle {
    pub position: Vector2<f32>,
    pub radius: f32,
    pub velocity: Vector2<f32>,
}

impl Obstacle {
    pub fn new(position: Vector2<f32>, radius: f32, velocity: Vector2<f32>) -> Self {
        Self { position, radius, velocity }
    }

    pub fn new_without_speed(position: Vector2<f32>, radius: f32) -> Self {
        Self { position, radius, velocity: Vector2::new(0.0, 0.0) }
    }
}

pub fn compute_path(
    start: &mut Vector2<f32>,
    goal: &Vector2<f32>,
    obstacles: &[Obstacle],
    start_speed: Option<Vector2<f32>>,
) -> [Vector2<f32>; N_POINTS] {
    mod algo_constants {
        pub const ALPHA: f32 = 0.00001;
        pub const BETA: f32 = 0.01;
        pub const INFLUENCE_FACTOR: (usize, usize) = (5, 1);
    }

    let mut path: [Vector2<f32>; N_POINTS] = [Vector2::new(0.0, 0.0); N_POINTS];
    let start_speed = start_speed.unwrap_or(Vector2::new(0.0, 0.0));

    use algo_constants::*;

    for index in 0..N_POINTS {
        let mut f:Vector2<f32> = (*goal - *start).normalize();
        let dist = (goal - *start).norm();
        let attractive_force = ALPHA * dist;
        f *= attractive_force;

        let (base_factor, speed_factor) = INFLUENCE_FACTOR;
        for obs in obstacles {
            let d = (obs.position - *start).norm();
            let relative_speed = (*start - obs.position).normalize()
                .dot(&(start_speed - obs.velocity)).abs();
            let influence_radius = base_factor as f32 * obs.radius + relative_speed * speed_factor as f32;
            if d < influence_radius {
                let repulsive_force = (1.0 / (d - 2.0 * obs.radius) - 1.0 / (influence_radius - 2.0 * obs.radius))
                    * (relative_speed * BETA + 1.0);
                f += (*start - obs.position).normalize() * repulsive_force;
            }
        }

        *start += f.normalize() * POINT_SPACING;


        if index == 0 || (goal - *start).norm() < (goal - path[index - 1]).norm() {
            path[index] = start.clone();
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
        let mut start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(10.0, 10.0);
        let direction = (goal - start).normalize();
        let expected_points: [Vector2<f32>; 5] = [
            start + direction * 0.1,
            start + direction * 0.2,
            start + direction * 0.3,
            start + direction * 0.4,
            start + direction * 0.5
        ];

        let path = compute_path(&mut start, &goal, &[], None);
        for i in 0..N_POINTS {
            assert!(approx_eq!(f32, (path[i]- expected_points[i]).norm(), 0.0 , epsilon = 1e-6));
        }
    }

    // TODO: add more tests, including tests with obstacles
}
