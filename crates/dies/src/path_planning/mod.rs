extern crate nalgebra as na;
use std::error::Error;
use na::Vector2;
extern crate float_cmp;
use float_cmp::approx_eq;
extern crate plotters;
use plotters::prelude::*;

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
        pub const ALPHA: f32 = 0.00000001;
        pub const BETA: f32 = 0.00001;
        pub const INFLUENCE_FACTOR: (f32, f32) = (0.005, 0.001);
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

pub fn plot_environment(start: &mut Vector2<f32>, goal: &Vector2<f32>, obstacles: &[Obstacle]) -> Result<(), Box<dyn Error>> {
    let root_area = BitMapBackend::new("plot.png", (900, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .build_cartesian_2d(0.0..6.0, 0.0..9.0)?;

    chart.draw_series(vec![
        Circle::new((start.x as f64, start.y as f64), 3, &RED),
        Circle::new((goal.x as f64, goal.y as f64), 3, &GREEN),
    ])?;

    for obstacle in obstacles {
        println!("Obstacle: {:?}", obstacle.position);
        chart.draw_series(vec![
            Circle::new((obstacle.position.x as f64, obstacle.position.y as f64), (obstacle.radius*100.0) as f64, &BLACK)
        ])?;
    }

    let path = compute_path(start, &goal, obstacles, None);
    let path_f64: Vec<(f64, f64)> = path.iter().map(|point| (point.x as f64, point.y as f64)).collect();
    chart.draw_series(LineSeries::new(path_f64, &BLUE))?;

    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_obstacles() {
        let mut start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(9.0, 6.0);
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

    #[test]
    fn test_1_obstacle_middle() {
        let mut start = Vector2::new(0.0, 0.0);
        let goal = Vector2::new(3.0, 5.0);
        let obstacles = [Obstacle::new(Vector2::new(1.5, 2.5), 0.08,
                                       Vector2::new(1.0, 0.0))];

        let path = compute_path(&mut start, &goal, &obstacles, None);

        let mut last_distance_to_goal = (goal - Vector2::new(0.0, 0.0)).norm();

        for point in &path {

            let distance_to_goal = (goal - *point).norm();
            assert!(distance_to_goal <= last_distance_to_goal + 1e-6);
            last_distance_to_goal = distance_to_goal;
            assert!((point - obstacles[0].position).norm() > obstacles[0].radius);
        }
    }

    /*
    #[test]
    fn test_with_plot()  -> Result<(), Box<dyn std::error::Error>>{
        let mut start = Vector2::new(1.0, 1.0);
        let goal = Vector2::new(3.0, 5.0);
        let obstacles = [Obstacle::new(Vector2::new(2.0, 2.5), 0.08,
                                       Vector2::new(1.0, 0.0))];

        plot_environment(&mut start, &goal, &obstacles)?;
        Ok(())
    }*/

}
