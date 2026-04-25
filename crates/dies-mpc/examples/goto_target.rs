//! Single-robot goto-target sanity example. Prints the planned trajectory as
//! TSV to stdout.
//!
//! Run with: `cargo run -p dies-mpc --example goto_target > traj.tsv`
//! Plot `px` vs `py`, or use the extra columns to inspect vx/vy and command.

use dies_mpc::{solve, MpcTarget, RobotParams, RobotState, SolverConfig, Vec2};

fn main() {
    let target = MpcTarget::goto(Vec2::new(2000.0, 0.0));
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
    let result = solve(state, &headings, &target, &params, None, &cfg);

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
