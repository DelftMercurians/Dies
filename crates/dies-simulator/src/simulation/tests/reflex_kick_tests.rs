use super::*;

/// Build a sim with one blue robot at the origin facing +x and a ball seated
/// in its dribbler, then apply `cmd` and step a few frames.
fn sim_with_ball_at_dribbler(cmd: PlayerGlobalMoveCmd) -> Simulation {
    let mut sim = SimulationBuilder::new(SimulationConfig::default())
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

fn ball_vx(sim: &Simulation) -> f64 {
    let handle = sim.physics.ball.as_ref().unwrap()._rigid_body_handle;
    sim.physics.rigid_body_set.get(handle).unwrap().linvel().x
}

#[test]
fn reflex_kick_fires_on_ball_contact() {
    // ARM_REFLEX_KICK with no counter increment must fire once the ball is at
    // the breakbeam — the strike-through restart release.
    let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
    cmd.heading_setpoint = 0.0;
    cmd.robot_cmd = RobotCmd::ArmReflex;
    let sim = sim_with_ball_at_dribbler(cmd);
    assert!(
        ball_vx(&sim) > 100.0,
        "reflex kick should have driven the ball forward, vx={}",
        ball_vx(&sim)
    );
}

#[test]
fn smart_kick_fires_on_counter_increment() {
    // Unchanged original path: ARM_COUNTER_KICK (RobotCmd::Arm) fires when the
    // kick counter increments past the robot's last seen value.
    let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
    cmd.heading_setpoint = 0.0;
    cmd.robot_cmd = RobotCmd::Arm;
    cmd.kick_counter = 1;
    let sim = sim_with_ball_at_dribbler(cmd);
    assert!(
        ball_vx(&sim) > 100.0,
        "smart kick should have driven the ball forward, vx={}",
        ball_vx(&sim)
    );
}

#[test]
fn no_kick_without_trigger() {
    // Plain arm with no counter change and no reflex must not kick.
    let mut cmd = PlayerGlobalMoveCmd::zero(PlayerId::new(0));
    cmd.heading_setpoint = 0.0;
    cmd.robot_cmd = RobotCmd::Arm;
    let sim = sim_with_ball_at_dribbler(cmd);
    assert!(
        ball_vx(&sim).abs() < 50.0,
        "ball should be (near) stationary without a kick, vx={}",
        ball_vx(&sim)
    );
}
