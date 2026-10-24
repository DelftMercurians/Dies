use super::*;

fn sim_with_ball() -> Simulation {
    SimulationBuilder::new(SimulationConfig::default())
        .add_ball(Vector::new(0.0, 0.0, BALL_RADIUS))
        .build()
}

#[test]
fn ball_inside_real_field_is_not_out() {
    // Regression: a corner free-kick ball resting ~62 mm inside the goal line
    // (the exact stuck position from the log) must NOT count as out. The old
    // 100 mm test boundary shrank the field below the 200 mm placement margin,
    // so it flagged a legally-placed corner free kick as out and instantly
    // re-stopped the game.
    let mut sim = sim_with_ball();
    sim.place_ball(-4437.86, 2738.58);
    assert!(!sim.ball_out(), "ball inside the real field flagged as out");
}

#[test]
fn placed_free_kick_ball_is_never_out() {
    // Any ball snapped to a valid free-kick spot sits within the placement
    // margin and must stay in play.
    let mut sim = sim_with_ball();
    for desired in [
        Vector2::new(-4300.0, 2900.0), // top-left corner
        Vector2::new(4300.0, -2900.0), // bottom-right corner
        Vector2::new(4400.0, 0.0),
    ] {
        let p = sim.valid_free_kick_position(desired);
        sim.place_ball(p.x, p.y);
        assert!(
            !sim.ball_out(),
            "placed free-kick ball {p:?} flagged as out"
        );
    }
}

#[test]
fn ball_past_the_line_is_out() {
    let mut sim = sim_with_ball();
    sim.place_ball(-4600.0, 0.0); // past the goal line
    assert!(sim.ball_out(), "ball past the goal line not flagged out");
    sim.place_ball(0.0, 3100.0); // past the touch line
    assert!(sim.ball_out(), "ball past the touch line not flagged out");
}

fn set_ball_vel(sim: &mut Simulation, vx: f64, vy: f64) {
    let h = sim.physics.ball.as_ref().unwrap()._rigid_body_handle;
    sim.physics
        .rigid_body_set
        .get_mut(h)
        .unwrap()
        .set_linvel(Vector::new(vx, vy, 0.0), true);
}

fn ball_xy(sim: &Simulation) -> Vector2 {
    let h = sim.physics.ball.as_ref().unwrap()._rigid_body_handle;
    sim.physics
        .rigid_body_set
        .get(h)
        .unwrap()
        .position()
        .translation
        .vector
        .xy()
}

/// Regression for the botched free-kick sequence (log frame 813): a ball that
/// rolls out of bounds while in `Run` must be teleported back to a legal
/// free-kick spot *and stay there* through the Stop + FreeKick period, so the
/// free kick can actually be taken. The original bug let the ball keep rolling
/// out to the boundary wall; the free kick was awarded with the ball still out,
/// `check_ball_in_play` mistook the ball's drift along the wall for the kick
/// being taken, and the sequence collapsed back into a force-start.
#[test]
fn rolling_ball_out_is_relocated_in_bounds_for_free_kick() {
    let mut sim = sim_with_ball();
    // Last toucher set so a free kick (not a neutral force-start) is awarded.
    sim.referee.last_touch_info = Some((PlayerId::new(0), TeamColor::Blue));
    sim.referee.game_state = SimulationGameState::run();

    // Seat the ball near the negative goal line, still in bounds, flying out
    // fast and at an angle (mirrors the logged corner roll-out).
    sim.place_ball(-4400.0, -1700.0);
    set_ball_vel(&mut sim, -1500.0, 350.0);

    let hl = sim.config.field_geometry.field_length / 2.0;
    let hw = sim.config.field_geometry.field_width / 2.0;

    // Step through the whole restart at 60 Hz: Stop (2 s) + free kick, with
    // enough headroom for the 10 s free-kick timeout to resume play since this
    // test has no kicker robot. Once the autoref has awarded the free kick
    // (StopAndFreeKick / FreeKick), the ball must be sitting at a legal
    // in-bounds spot — never pinned against the boundary wall.
    let mut saw_free_kick = false;
    let mut reached_run_after = false;
    for _ in 0..780 {
        sim.step(1.0 / 60.0);
        let b = ball_xy(&sim);
        match sim.referee.game_state {
            SimulationGameState::StopAndFreeKick { .. } | SimulationGameState::FreeKick { .. } => {
                saw_free_kick = true;
                assert!(
                    b.x.abs() <= hl + 1e-3 && b.y.abs() <= hw + 1e-3,
                    "free kick awarded with the ball still out of bounds: {b:?}"
                );
            }
            SimulationGameState::Run { .. } if saw_free_kick => {
                reached_run_after = true;
            }
            // The free-kick restart must not collapse back into a neutral
            // force-start because the ball was left out of bounds.
            SimulationGameState::StopAndForceStart { .. } if saw_free_kick => {
                panic!("free-kick restart collapsed into a force-start (ball left out)");
            }
            _ => {}
        }
    }

    assert!(saw_free_kick, "free kick was never awarded");
    assert!(
        reached_run_after,
        "game never resumed (Run) after the free kick"
    );
    let b = ball_xy(&sim);
    assert!(
        b.x.abs() <= hl && b.y.abs() <= hw,
        "ball still out of bounds at end of restart: {b:?}"
    );
}
