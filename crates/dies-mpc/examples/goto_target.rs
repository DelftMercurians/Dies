//! Quick-look sanity example: solve a single-robot `GoToPos` with one
//! circular obstacle slightly off the direct line, then print the
//! resulting trajectory as TSV to stdout.
//!
//! Run with: `cargo run -p dies-mpc --example goto_target > traj.tsv`
//! Then plot `px` vs `py`, or use the extra columns to inspect vx/vy.

use dies_mpc::{
    solve, CostWeights, FieldBounds, MpcTarget, ObstacleShape, PredictedObstacle,
    ReferenceTrajectory, RobotParams, RobotState, SolverConfig, TerminalMode, Vec2, WorldSnapshot,
};

fn main() {
    let target = Vec2::new(3000.0, 0.0);
    let mpc_target = MpcTarget {
        reference: ReferenceTrajectory::StaticPoint(target),
        terminal: TerminalMode::Position { p: target },
        weights: CostWeights::default(),
        care: 1.0,
        aggressiveness: 0.3,
    };
    let world = WorldSnapshot {
        obstacles: vec![PredictedObstacle {
            shape: ObstacleShape::Circle {
                center: Vec2::new(1500.0, 80.0),
                radius: 150.0,
            },
            velocity: Vec2::zeros(),
            safe_dist: 200.0,
            no_cost_dist: 500.0,
            weight_scale: 2.0,
        }],
        field_bounds: FieldBounds::centered(9000.0, 6000.0, 1000.0, 2000.0),
    };
    let params = RobotParams::default_hand_tuned();
    let cfg = SolverConfig {
        horizon: 25,
        ..SolverConfig::default()
    };
    let state = RobotState {
        pos: Vec2::zeros(),
        vel: Vec2::zeros(),
    };
    let headings = vec![0.0; cfg.horizon + 1];
    let result = solve(state, &headings, &mpc_target, &params, &world, None, &cfg);

    eprintln!(
        "# solve: iters={} converged={} cost={:.4} time={}us",
        result.iters, result.converged, result.final_cost, result.solve_time_us
    );
    println!("k\tt\tpx\tpy\tvx\tvy\tux\tuy");
    for (k, s) in result.trajectory.states.iter().enumerate() {
        let t = k as f64 * cfg.dt;
        let (ux, uy) = if k < result.trajectory.controls.len() {
            let u = result.trajectory.controls[k];
            (u.x, u.y)
        } else {
            (f64::NAN, f64::NAN)
        };
        println!(
            "{}\t{:.3}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{:.1}",
            k, t, s[0], s[1], s[2], s[3], ux, uy
        );
    }
}
