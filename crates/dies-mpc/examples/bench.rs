//! Benchmarks for `dies_mpc::solve` across realistic SSL scenarios.
//!
//! Run with release optimizations: `cargo run --release -p dies-mpc --example bench`.
//! Debug-mode numbers are not meaningful for control-loop budget analysis.

use std::hint::black_box;
use std::time::Instant;

use dies_mpc::{
    solve, CostWeights, FieldBounds, MpcTarget, ObstacleShape, PredictedObstacle,
    ReferenceTrajectory, RobotParams, RobotState, SolveResult, SolverConfig, TerminalMode,
    Trajectory, Vec2, WorldSnapshot,
};

struct Run {
    wall_us: u64,
    result: SolveResult,
}

struct Stats {
    min_us: u64,
    median_us: u64,
    p95_us: u64,
    max_us: u64,
    mean_iters: f64,
    converged_frac: f64,
}

fn summarize(runs: &[Run]) -> Stats {
    let mut times: Vec<u64> = runs.iter().map(|r| r.wall_us).collect();
    times.sort_unstable();
    let n = times.len();
    let idx = |q: f64| ((n - 1) as f64 * q).round() as usize;
    Stats {
        min_us: times[0],
        median_us: times[idx(0.5)],
        p95_us: times[idx(0.95)],
        max_us: times[n - 1],
        mean_iters: runs.iter().map(|r| r.result.iters as f64).sum::<f64>() / n as f64,
        converged_frac: runs.iter().filter(|r| r.result.converged).count() as f64 / n as f64,
    }
}

fn print_row(label: &str, s: &Stats) {
    println!(
        "{:<40}  min={:>5}µs  med={:>5}µs  p95={:>5}µs  max={:>6}µs  iters={:.1}  conv={:.0}%",
        label,
        s.min_us,
        s.median_us,
        s.p95_us,
        s.max_us,
        s.mean_iters,
        100.0 * s.converged_frac
    );
}

fn ssl_field() -> FieldBounds {
    // Division B: 9000×6000 mm playing area, 1000 mm penalty depth.
    FieldBounds::centered(9000.0, 6000.0, 1000.0, 2000.0)
}

fn target_goto(p: Vec2, care: f64, aggro: f64) -> MpcTarget {
    MpcTarget {
        reference: ReferenceTrajectory::StaticPoint(p),
        terminal: TerminalMode::Position { p },
        weights: CostWeights::default(),
        care,
        aggressiveness: aggro,
    }
}

fn robot_obstacle(center: Vec2, vel: Vec2) -> PredictedObstacle {
    PredictedObstacle {
        shape: ObstacleShape::Circle {
            center,
            radius: 90.0,
        },
        velocity: vel,
        safe_dist: 90.0,
        no_cost_dist: 300.0,
        weight_scale: 1.0,
    }
}

/// Deterministic jittered starting conditions. Returns N configurations.
fn scenarios(
    n_samples: usize,
    obstacle_count: usize,
) -> Vec<(RobotState, MpcTarget, WorldSnapshot)> {
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = i as f64;
        // Start somewhere on the near half of the field, target on the far half.
        let start = Vec2::new(
            -2500.0 + (t * 137.0).rem_euclid(4000.0) - 2000.0,
            -1500.0 + (t * 91.0).rem_euclid(3000.0) - 1500.0,
        );
        let target = Vec2::new(
            1500.0 + (t * 73.0).rem_euclid(2500.0) - 1250.0,
            -1500.0 + (t * 109.0).rem_euclid(3000.0) - 1500.0,
        );
        let v0 = Vec2::new(
            (t * 41.0).rem_euclid(1000.0) - 500.0,
            (t * 23.0).rem_euclid(1000.0) - 500.0,
        );
        let mut obstacles = Vec::with_capacity(obstacle_count);
        for j in 0..obstacle_count {
            let s = t + j as f64 * 17.0;
            let ox = -3500.0 + (s * 31.0).rem_euclid(7000.0);
            let oy = -2000.0 + (s * 19.0).rem_euclid(4000.0);
            let vx = (s * 7.0).rem_euclid(800.0) - 400.0;
            let vy = (s * 11.0).rem_euclid(800.0) - 400.0;
            obstacles.push(robot_obstacle(Vec2::new(ox, oy), Vec2::new(vx, vy)));
        }
        let world = WorldSnapshot {
            obstacles,
            field_bounds: ssl_field(),
        };
        out.push((
            RobotState {
                pos: start,
                vel: v0,
            },
            target_goto(target, 1.0, 0.3),
            world,
        ));
    }
    out
}

fn time_solve(
    state: RobotState,
    headings: &[f64],
    target: &MpcTarget,
    params: &RobotParams,
    world: &WorldSnapshot,
    warm: Option<&Trajectory>,
    cfg: &SolverConfig,
) -> Run {
    // Each sample: warm up the cache with one untimed call, then time a fresh
    // solve from the same inputs. Wall-clock external timing covers *both*
    // internal init paths of `solve()`.
    let _warmup = black_box(solve(
        black_box(state),
        black_box(headings),
        black_box(target),
        black_box(params),
        black_box(world),
        warm,
        black_box(cfg),
    ));
    let start = Instant::now();
    let result = solve(
        black_box(state),
        black_box(headings),
        black_box(target),
        black_box(params),
        black_box(world),
        warm,
        black_box(cfg),
    );
    let wall_us = start.elapsed().as_micros() as u64;
    Run {
        wall_us,
        result: black_box(result),
    }
}

fn bench_cold(
    label: &str,
    cfg: &SolverConfig,
    params: &RobotParams,
    scenarios: &[(RobotState, MpcTarget, WorldSnapshot)],
) {
    let headings = vec![0.0; cfg.horizon + 1];
    let mut runs = Vec::with_capacity(scenarios.len());
    for (state, target, world) in scenarios {
        runs.push(time_solve(
            *state, &headings, target, params, world, None, cfg,
        ));
    }
    let s = summarize(&runs);
    print_row(label, &s);
}

fn bench_warm(
    label: &str,
    cfg: &SolverConfig,
    params: &RobotParams,
    scenarios: &[(RobotState, MpcTarget, WorldSnapshot)],
) {
    let headings = vec![0.0; cfg.horizon + 1];
    let mut runs = Vec::with_capacity(scenarios.len());
    for (state, target, world) in scenarios {
        let cold = solve(*state, &headings, target, params, world, None, cfg);
        let warm_start = cold.trajectory.clone();
        runs.push(time_solve(
            *state,
            &headings,
            target,
            params,
            world,
            Some(&warm_start),
            cfg,
        ));
    }
    let s = summarize(&runs);
    print_row(label, &s);
}

fn bench_full_frame(
    label: &str,
    cfg: &SolverConfig,
    params: &RobotParams,
    robots_per_frame: usize,
    obstacle_count: usize,
    frames: usize,
) {
    // Six independent robots per frame, measured sequentially. Reports
    // *total frame time* statistics — the metric that matters for the
    // 16.7 ms budget at 60 Hz.
    let headings = vec![0.0; cfg.horizon + 1];
    let scens = scenarios(robots_per_frame * frames, obstacle_count);
    let mut warm_trajs: Vec<Option<Trajectory>> = vec![None; robots_per_frame];
    let mut frame_us = Vec::with_capacity(frames);
    let mut iter_sum = 0u32;
    let mut iter_count = 0u32;
    for f in 0..frames {
        let start = Instant::now();
        #[allow(clippy::needless_range_loop)]
        for r in 0..robots_per_frame {
            let idx = f * robots_per_frame + r;
            let (state, target, world) = &scens[idx];
            let res = solve(
                black_box(*state),
                black_box(&headings),
                black_box(target),
                black_box(params),
                black_box(world),
                warm_trajs[r].as_ref(),
                black_box(cfg),
            );
            iter_sum += res.iters;
            iter_count += 1;
            warm_trajs[r] = Some(black_box(res.trajectory));
        }
        frame_us.push(start.elapsed().as_micros() as u64);
    }
    frame_us.sort_unstable();
    let n = frame_us.len();
    let idx = |q: f64| ((n - 1) as f64 * q).round() as usize;
    println!(
        "{:<40}  min={:>5}µs  med={:>5}µs  p95={:>5}µs  max={:>6}µs  mean_iters={:.1}",
        label,
        frame_us[0],
        frame_us[idx(0.5)],
        frame_us[idx(0.95)],
        frame_us[n - 1],
        iter_sum as f64 / iter_count as f64,
    );
}

fn main() {
    let params = RobotParams::default_hand_tuned();
    let cfg_default = SolverConfig::default();

    let n_scen = 200;

    println!("=== per-robot solve time ===");
    println!("(SolverConfig::default(): horizon=10, dt=60ms, max_iters=15)");
    println!();

    let s0 = scenarios(n_scen, 0);
    let s3 = scenarios(n_scen, 3);
    let s6 = scenarios(n_scen, 6);
    let s10 = scenarios(n_scen, 10);

    bench_cold("cold, 0 obstacles", &cfg_default, &params, &s0);
    bench_cold("cold, 3 obstacles", &cfg_default, &params, &s3);
    bench_cold("cold, 6 obstacles", &cfg_default, &params, &s6);
    bench_cold("cold, 10 obstacles", &cfg_default, &params, &s10);
    println!();
    bench_warm("warm, 0 obstacles", &cfg_default, &params, &s0);
    bench_warm("warm, 3 obstacles", &cfg_default, &params, &s3);
    bench_warm("warm, 6 obstacles", &cfg_default, &params, &s6);
    bench_warm("warm, 10 obstacles", &cfg_default, &params, &s10);

    println!();
    println!("=== horizon sensitivity (6 obstacles, cold) ===");
    for horizon in [10usize, 15, 20, 30, 50] {
        let cfg = SolverConfig {
            horizon,
            ..cfg_default.clone()
        };
        bench_cold(
            &format!("horizon={}", horizon),
            &cfg,
            &params,
            &scenarios(n_scen, 6),
        );
    }

    println!();
    println!("=== 6-robot sequential frame budget ===");
    bench_full_frame(
        "6 robots × 3 obstacles each, 100 frames",
        &cfg_default,
        &params,
        6,
        3,
        100,
    );
    bench_full_frame(
        "6 robots × 6 obstacles each, 100 frames",
        &cfg_default,
        &params,
        6,
        6,
        100,
    );
    bench_full_frame(
        "6 robots × 10 obstacles each, 100 frames",
        &cfg_default,
        &params,
        6,
        10,
        100,
    );
}
