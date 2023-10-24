use nalgebra::Vector2;

/// The number of points in the path
const N_POINTS: usize = 5;

/// The spacing between points in the path in meters
const POINT_SPACING: f32 = 0.1;

/// A circular obstacle
pub struct Obstacle {
    pub position: Vector2<f32>,
    pub radius: f32,
}

impl Obstacle {
    pub fn new(position: Vector2<f32>, radius: f32) -> Self {
        Self { position, radius }
    }
}

pub fn compute_path(
    start: &Vector2<f32>,
    goal: &Vector2<f32>,
    obstacles: &[Obstacle],
) -> [Vector2<f32>; N_POINTS] {
    // TODO: implement path planning
    [Vector2::new(0.0, 0.0); N_POINTS]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_obstacles() {
        let start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(10.0, 10.0);

        let path = compute_path(&start, &goal, &[]);
        for i in 1..(N_POINTS + 1) {
            assert_eq!(
                path[0],
                start + Vector2::new(1.0, 1.0).normalize() * POINT_SPACING * i as f32
            );
        }
    }

    // TODO: add more tests, including tests with obstacles
}
