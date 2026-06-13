//! Golden-file dump for the Python parity test.
//!
//! Runs the iLQR solver on a handful of representative scenarios and dumps
//! the inputs + outputs into `codegen/pysim/tests/golden.json`. The Python
//! test `pysim/tests/test_parity.py` loads this file and asserts numerical
//! agreement between the Rust and Python solvers.
//!
//! Regenerate after intentional changes to `solver.rs`:
//!
//! ```bash
//! cargo test -p dies-mpc --test dump_golden -- --ignored --nocapture
//! ```
//!
//! Gated behind `#[ignore]` so it doesn't run in the default `cargo test`
//! suite — `git diff` should never fire on every test run.

use std::fs;
use std::path::PathBuf;

use dies_mpc::solver::solve;
use dies_mpc::types::{
    CostWeights, MpcTarget, RobotParams, RobotState, SolverConfig, Trajectory, Vec2,
};
use serde_json::{json, Value};

fn weights_to_json(w: &CostWeights) -> Value {
    json!({
        "position":            w.position,
        "velocity":            w.velocity,
        "control":             w.control,
        "control_smoothness":  w.control_smoothness,
        "heading":             w.heading,
        "heading_control":     w.heading_control,
    })
}

fn cfg_to_json(c: &SolverConfig) -> Value {
    json!({
        "horizon":     c.horizon,
        "dt":          c.dt,
        "max_iters":   c.max_iters,
        "cost_tol":    c.cost_tol,
        "reg_init":    c.reg_init,
        "reg_min":     c.reg_min,
        "reg_max":     c.reg_max,
        "reg_factor":  c.reg_factor,
    })
}

fn target_to_json(t: &MpcTarget) -> Value {
    json!({
        "p":       [t.p.x, t.p.y],
        "v":       [t.v.x, t.v.y],
        "heading": t.heading,
        "weights": weights_to_json(&t.weights),
    })
}

fn params_to_json(p: &RobotParams) -> Value {
    json!({
        "tau":       p.tau,
        "accel_max": p.accel_max,
        "tau_yaw":   p.tau_yaw,
        "omega_max": p.omega_max,
    })
}

fn state_to_json(s: &RobotState) -> Value {
    json!({
        "pos":     [s.pos.x, s.pos.y],
        "vel":     [s.vel.x, s.vel.y],
        "heading": s.heading,
    })
}

fn traj_to_json(t: &Trajectory) -> Value {
    let states: Vec<[f64; 5]> = t
        .states
        .iter()
        .map(|s| [s[0], s[1], s[2], s[3], s[4]])
        .collect();
    let controls: Vec<[f64; 3]> = t.controls.iter().map(|u| [u[0], u[1], u[2]]).collect();
    json!({ "states": states, "controls": controls })
}

struct Case {
    name: &'static str,
    state: RobotState,
    target: MpcTarget,
    params: RobotParams,
    cfg: SolverConfig,
    warm_start: Option<Trajectory>,
}

fn run_case(case: &Case) -> Value {
    let result = solve(
        case.state,
        &case.target,
        &case.params,
        case.warm_start.as_ref(),
        &case.cfg,
    );

    json!({
        "name": case.name,
        "inputs": {
            "state":        state_to_json(&case.state),
            "target":       target_to_json(&case.target),
            "params":       params_to_json(&case.params),
            "cfg":          cfg_to_json(&case.cfg),
            "warm_start":   case.warm_start.as_ref().map(traj_to_json),
        },
        "outputs": {
            "trajectory":  traj_to_json(&result.trajectory),
            "final_cost":  result.final_cost,
            "iters":       result.iters,
            "converged":   result.converged,
        },
    })
}

#[test]
#[ignore = "regenerates the golden JSON for pysim parity tests; run explicitly"]
fn dump_golden() {
    let p = RobotParams::default_hand_tuned();
    let cfg_default = SolverConfig::default();
    let cfg_short = SolverConfig {
        horizon: 50,
        ..SolverConfig::default()
    };

    // Scenario 1: solver should produce ~zero control at target.
    let target_a = MpcTarget::goto(Vec2::new(1000.0, 500.0));
    let case_at_rest = Case {
        name: "at_rest_at_target",
        state: RobotState {
            pos: target_a.p,
            vel: Vec2::zeros(),
            heading: 0.0,
        },
        target: target_a,
        params: p.clone(),
        cfg: cfg_short.clone(),
        warm_start: None,
    };

    // Scenario 2: straight-line goto from origin (matches existing example).
    let case_goto = Case {
        name: "goto_2000_0_short",
        state: RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        },
        target: MpcTarget::goto(Vec2::new(2000.0, 0.0)),
        params: p.clone(),
        cfg: cfg_short.clone(),
        warm_start: None,
    };

    // Scenario 3: off-axis target with a nonzero start/desired heading —
    // exercises full 2-D dynamics plus the heading lag.
    let mut target_off = MpcTarget::goto(Vec2::new(1500.0, -800.0));
    target_off.heading = 0.9;
    let case_off_axis = Case {
        name: "goto_1500_-800_short",
        state: RobotState {
            pos: Vec2::new(-200.0, 100.0),
            vel: Vec2::new(300.0, -100.0),
            heading: -0.4,
        },
        target: target_off,
        params: p.clone(),
        cfg: cfg_short.clone(),
        warm_start: None,
    };

    // Scenario 4: default horizon (matches production cfg).
    let case_default_horizon = Case {
        name: "goto_1500_0_default_horizon",
        state: RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        },
        target: MpcTarget::goto(Vec2::new(1500.0, 0.0)),
        params: p.clone(),
        cfg: cfg_default.clone(),
        warm_start: None,
    };

    // Scenario 5: warm start derived from scenario 4's result.
    let warm_seed = solve(
        case_default_horizon.state,
        &case_default_horizon.target,
        &case_default_horizon.params,
        None,
        &case_default_horizon.cfg,
    );
    let case_warm = Case {
        name: "goto_1500_0_warm",
        state: RobotState {
            pos: Vec2::zeros(),
            vel: Vec2::zeros(),
            heading: 0.0,
        },
        target: MpcTarget::goto(Vec2::new(1500.0, 0.0)),
        params: p.clone(),
        cfg: cfg_default.clone(),
        warm_start: Some(warm_seed.trajectory),
    };

    let cases = [
        run_case(&case_at_rest),
        run_case(&case_goto),
        run_case(&case_off_axis),
        run_case(&case_default_horizon),
        run_case(&case_warm),
    ];

    let doc = json!({
        "_comment": "Generated by `cargo test -p dies-mpc --test dump_golden -- --ignored`",
        "cases": cases,
    });

    // Target path: crates/dies-mpc/codegen/pysim/tests/golden.json
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("codegen");
    path.push("pysim");
    path.push("tests");
    fs::create_dir_all(&path).expect("create pysim/tests dir");
    path.push("golden.json");
    fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).expect("write golden.json");
    eprintln!("wrote {} cases to {}", cases.len(), path.display());
}
