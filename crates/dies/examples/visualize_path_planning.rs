use anyhow::{Ok, Result};
use dies::path_planning::{compute_path, Obstacle};
use nalgebra::Vector2;
use plotters::prelude::*;

pub fn plot_environment(
    start: &Vector2<f32>,
    goal: &Vector2<f32>,
    obstacles: &[Obstacle],
) -> Result<()> {
    let root_area = BitMapBackend::new("plot.png", (900, 600)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart =
        ChartBuilder::on(&root_area).build_cartesian_2d(-1500.0..1500.0, -2000.0..2000.0)?;

    chart.draw_series(vec![
        Circle::new((start.x as f64, start.y as f64), 3, &RED),
        Circle::new((goal.x as f64, goal.y as f64), 3, &GREEN),
    ])?;

    for obstacle in obstacles {
        println!("Obstacle: {:?}", obstacle.position);
        chart.draw_series(vec![Circle::new(
            (obstacle.position.x as f64, obstacle.position.y as f64),
            (obstacle.radius * 0.1) as f64,
            &BLACK,
        )])?;
    }

    let mut path = compute_path(start, &goal, obstacles, None).to_vec();
    // Insert start to path
    path.insert(0, start.clone());
    let path_f64: Vec<(f64, f64)> = path
        .iter()
        .map(|point| (point.x as f64, point.y as f64))
        .collect();
    chart.draw_series(LineSeries::new(path_f64, &BLUE))?;

    // Draw the points making up the path
    for point in path {
        chart.draw_series(vec![Circle::new(
            (point.x as f64, point.y as f64),
            1,
            &BLUE,
        )])?;
    }

    Ok(())
}

fn main() -> Result<()> {
    // No obstacles
    // let start = Vector2::new(0.0, 0.0);
    // let goal = Vector2::new(9000.0, 6000.0);
    // let obstacles = [];

    // 1 obstacle in the middle
    let start = Vector2::new(0.0, 0.0);
    let goal = Vector2::new(3000.0, 5000.0);
    let obstacles = [Obstacle::new(
        Vector2::new(400.0, 200.0),
        80.0,
        Vector2::new(0.0, 0.0),
    )];

    // 2 obstacles close to start
    // let start = Vector2::new(0.0, 0.0);
    // let goal = Vector2::new(1000.0, 3000.0);
    // let obstacles = [
    //     Obstacle::new_without_vel(Vector2::new(500.0, 0.0), 250.0),
    //     Obstacle::new_without_vel(Vector2::new(0.0, 700.0), 250.0),
    // ];

    plot_environment(&start, &goal, &obstacles)?;
    Ok(())
}
