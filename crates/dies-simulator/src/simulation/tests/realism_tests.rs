use super::*;

/// Build a sim with one blue robot at the origin facing +x, a ball seated in
/// its dribbler, and the given realism config. Issue `cmd`, step, return the
/// sim.
fn sim_with(realism: RealismConfig, seed: u64, cmd: PlayerGlobalMoveCmd) -> Simulation {
    let mut config = SimulationConfig {
        seed,
        realism,
        ..Default::default()
    };
    config.blue_controlled = true;
    config.yellow_controlled = true;
    let mut sim = SimulationBuilder::new(config)
        .with_controlled_teams(true, true)
        .add_blue_player_with_id(0, Vector2::new(0.0, 0.0), Angle::from_radians(0.0))
        .add_ball(Vector::new(150.0, 0.0, BALL_RADIUS))
        .build();
    sim.push_global_cmd(TeamColor::Blue, cmd);
    for _ in 0..10 {
        sim.step(0.016);
    }
    sim
}

fn ball_vel(sim: &Simulation) -> Vector<f64> {
    let h = sim.physics.ball.as_ref().unwrap()._rigid_body_handle;
    *sim.physics.rigid_body_set.get(h).unwrap().linvel()
}

fn kick_cmd() -> PlayerGlobalMoveCmd {
    let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
    cmd.heading_setpoint = 0.0;
    cmd.robot_cmd = RobotCmd::ArmReflex;
    cmd
}

#[test]
fn shot_heading_noise_spreads_the_kick() {
    // With heading noise the kick direction wanders off the robot's facing, so
    // the lateral (vy) component is substantially nonzero across seeds; with no
    // noise it stays on the centerline (vy ≈ 0).
    let mut noisy = RealismConfig::default();
    noisy.teams = RealismTeams::Both;
    noisy.shot_heading_noise = 0.25;

    let mut sum_abs_vy_noisy = 0.0;
    let mut sum_abs_vy_clean = 0.0;
    let n = 24;
    for seed in 0..n {
        sum_abs_vy_noisy += ball_vel(&sim_with(noisy.clone(), seed, kick_cmd())).y.abs();
        sum_abs_vy_clean += ball_vel(&sim_with(RealismConfig::default(), seed, kick_cmd()))
            .y
            .abs();
    }
    let mean_noisy = sum_abs_vy_noisy / n as f64;
    let mean_clean = sum_abs_vy_clean / n as f64;
    assert!(
        mean_noisy > 200.0 && mean_noisy > 10.0 * mean_clean.max(1.0),
        "heading noise should spread the kick laterally: noisy |vy|={mean_noisy:.1} clean={mean_clean:.1}"
    );
}

#[test]
fn shot_failure_suppresses_the_kick() {
    // A guaranteed misfire leaves the ball on the dribbler.
    let mut r = RealismConfig::default();
    r.teams = RealismTeams::Both;
    r.shot_failure_prob = 1.0;
    let v = ball_vel(&sim_with(r, 7, kick_cmd()));
    assert!(
        v.norm() < 50.0,
        "misfire should not launch the ball, |v|={}",
        v.norm()
    );
}

#[test]
fn capture_failure_is_deterministic_and_team_scoped() {
    // capture_fail_prob is 0 for an untouched team and clamped to its max.
    let mut r = RealismConfig::default();
    r.teams = RealismTeams::Blue;
    r.capture_fail_base = 0.1;
    r.capture_fail_per_mps = 0.1;
    r.capture_fail_max = 0.45;
    assert_eq!(r.capture_fail_prob(TeamColor::Yellow, 5.0), 0.0);
    assert!((r.capture_fail_prob(TeamColor::Blue, 0.0) - 0.1).abs() < 1e-9);
    assert!((r.capture_fail_prob(TeamColor::Blue, 100.0) - 0.45).abs() < 1e-9);
}

#[test]
fn realism_preserves_determinism() {
    // Same (seed, realism) ⇒ identical kick outcome.
    let mut r = RealismConfig::default();
    r.teams = RealismTeams::Both;
    r.shot_heading_noise = 0.2;
    r.shot_power_noise = 0.2;
    let a = ball_vel(&sim_with(r.clone(), 3, kick_cmd()));
    let b = ball_vel(&sim_with(r, 3, kick_cmd()));
    assert_eq!(a.x.to_bits(), b.x.to_bits());
    assert_eq!(a.y.to_bits(), b.y.to_bits());
}
