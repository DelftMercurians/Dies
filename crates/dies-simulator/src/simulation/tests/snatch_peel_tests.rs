use super::*;

const BLUE: TeamColor = TeamColor::Blue;
const YELLOW: TeamColor = TeamColor::Yellow;

/// Blue holder at the origin facing +x with the ball seated in its dribbler,
/// and a yellow contester pressed against the ball from the far (+x) side.
fn setup() -> Simulation {
    SimulationBuilder::new(SimulationConfig::default())
        .with_controlled_teams(true, true)
        .add_blue_player_with_id(0, Vector2::new(0.0, 0.0), Angle::from_radians(0.0))
        .add_yellow_player_with_id(0, Vector2::new(340.0, 0.0), Angle::from_radians(PI))
        .add_ball(Vector::new(150.0, 0.0, BALL_RADIUS))
        .build()
}

fn hold(dribble: f64) -> PlayerMoveCmd {
    let mut c = PlayerMoveCmd::zero(PlayerId::new(0));
    c.dribble_speed = dribble;
    c
}

fn press(dribble: f64, w: f64) -> PlayerMoveCmd {
    let mut c = PlayerMoveCmd::zero(PlayerId::new(0));
    c.dribble_speed = dribble;
    c.w = w;
    c
}

fn holder(sim: &Simulation) -> Option<(PlayerId, TeamColor)> {
    sim.ball_interaction.ball_being_dribbled_by
}

fn ball_speed(sim: &Simulation) -> f64 {
    let h = sim.physics.ball.as_ref().unwrap()._rigid_body_handle;
    let v = sim.physics.rigid_body_set.get(h).unwrap().linvel();
    (v.x * v.x + v.y * v.y).sqrt()
}

/// A contester that presses *and rotates* peels the ball off the holder.
#[test]
fn rotating_press_strips_the_ball() {
    let mut sim = setup();

    // Phase A: blue establishes possession, yellow idle.
    for _ in 0..30 {
        sim.push_cmd(BLUE, hold(0.5));
        sim.step(1.0 / 60.0);
    }
    assert_eq!(
        holder(&sim),
        Some((PlayerId::new(0), BLUE)),
        "blue should hold the ball before the strip"
    );

    // Phase B: yellow presses with its dribbler on and rotates in place.
    let mut stripped_at = None;
    for f in 0..120 {
        sim.push_cmd(BLUE, hold(0.5));
        sim.push_cmd(YELLOW, press(0.8, 2.0));
        sim.step(1.0 / 60.0);
        if holder(&sim).is_none() {
            stripped_at = Some(f);
            break;
        }
    }

    let f = stripped_at.expect("rotating press should strip the ball within 2s");
    assert!(
        ball_speed(&sim) > 100.0,
        "stripped ball should be flung loose, speed={}",
        ball_speed(&sim)
    );
    // Slow peel, not instant: it should take a meaningful fraction of a second.
    assert!(f > 10, "strip happened suspiciously fast at frame {f}");
}

/// Pressing without rotating must NOT strip the ball — only ω peels.
#[test]
fn static_press_does_not_strip() {
    let mut sim = setup();
    for _ in 0..30 {
        sim.push_cmd(BLUE, hold(0.5));
        sim.step(1.0 / 60.0);
    }
    assert_eq!(holder(&sim), Some((PlayerId::new(0), BLUE)));

    for _ in 0..120 {
        sim.push_cmd(BLUE, hold(0.5));
        sim.push_cmd(YELLOW, press(0.8, 0.0)); // dribbler on, no rotation
        sim.step(1.0 / 60.0);
    }
    assert_eq!(
        holder(&sim),
        Some((PlayerId::new(0), BLUE)),
        "a non-rotating press should not peel the ball off the holder"
    );
}

/// Two robots reaching for a loose ball on the same frame (no prior holder)
/// must still trip the pop guard that squirts the ball out, rather than the
/// snatch peel path silently handing possession to one of them. The peel path
/// only applies once a robot has an *established* possession.
#[test]
fn fresh_5050_still_pops_loose() {
    let mut sim = setup();
    // Both reach for the ball on the same frame, neither rotating.
    let mut max_speed = 0.0_f64;
    for _ in 0..20 {
        sim.push_cmd(BLUE, hold(0.8));
        sim.push_cmd(YELLOW, press(0.8, 0.0));
        sim.step(1.0 / 60.0);
        max_speed = max_speed.max(ball_speed(&sim));
    }
    assert!(
        max_speed > 800.0,
        "a fresh 50/50 should pop the ball loose (guard fires); max speed={max_speed}"
    );
}
