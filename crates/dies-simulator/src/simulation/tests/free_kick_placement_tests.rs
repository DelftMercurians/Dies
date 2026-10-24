use super::*;

/// Distance from a point to an axis-aligned rectangle (0 if inside).
fn dist_to_rect(p: Vector2, min: Vector2, max: Vector2) -> f64 {
    let dx = (min.x - p.x).max(0.0).max(p.x - max.x);
    let dy = (min.y - p.y).max(0.0).max(p.y - max.y);
    (dx * dx + dy * dy).sqrt()
}

/// Assert a position obeys rule 5.3.3: inside the field margin and ≥1 m from
/// both defense areas.
fn assert_valid(sim: &Simulation, p: Vector2) {
    let g = &sim.config.field_geometry;
    let (hl, hw) = (g.field_length / 2.0, g.field_width / 2.0);
    let half_pa_w = g.penalty_area_width / 2.0;
    assert!(
        p.x.abs() <= hl - FREE_KICK_PLACEMENT_MARGIN + 1e-6
            && p.y.abs() <= hw - FREE_KICK_PLACEMENT_MARGIN + 1e-6,
        "outside field margin: {p:?}"
    );
    for &goal_x in &[-hl, hl] {
        let inner_x = goal_x - g.penalty_area_depth * goal_x.signum();
        let min = Vector2::new(goal_x.min(inner_x), -half_pa_w);
        let max = Vector2::new(goal_x.max(inner_x), half_pa_w);
        let d = dist_to_rect(p, min, max);
        assert!(
            d >= FREE_KICK_DEFENSE_AREA_DISTANCE - 1e-6,
            "too close to defense area: {p:?} d={d}"
        );
    }
}

#[test]
fn relocates_ball_jammed_against_defense_area() {
    // The exact position from the stuck free kick: 63 mm outside the
    // opponent defense-area corner, where the kicker could not get behind it.
    let sim = Simulation::default();
    let fixed = sim.valid_free_kick_position(Vector2::new(-4300.0, 1063.1));
    assert!(
        (fixed - Vector2::new(-4300.0, 1063.1)).norm() > 1.0,
        "should have moved the ball"
    );
    assert_valid(&sim, fixed);
}

#[test]
fn ball_out_over_goal_line_is_not_left_inside_the_box() {
    // Regression: a ball that left over the goal line beside the goal lands
    // inside the defense area, closer to the goal line than to the front edge
    // or the sides. The keep-out must push it out a *field-facing* face, not
    // back through the goal line (which the field clamp would then undo,
    // stranding the free kick at an illegal in-area position).
    let sim = Simulation::default();
    for desired in [
        Vector2::new(4300.0, 693.0),   // the exact stuck case from the log
        Vector2::new(-4300.0, -693.0), // mirror at the other goal
        Vector2::new(4400.0, 0.0),     // dead centre, hard against the goal line
    ] {
        let fixed = sim.valid_free_kick_position(desired);
        assert_valid(&sim, fixed);
    }
}

#[test]
fn leaves_already_valid_position_untouched() {
    let sim = Simulation::default();
    for desired in [
        Vector2::new(0.0, 0.0),
        Vector2::new(1500.0, -500.0),
        Vector2::new(-2000.0, 2000.0),
    ] {
        let fixed = sim.valid_free_kick_position(desired);
        assert!(
            (fixed - desired).norm() < 1e-6,
            "moved valid pos {desired:?}"
        );
        assert_valid(&sim, fixed);
    }
}

#[test]
fn relocates_near_both_goals() {
    let sim = Simulation::default();
    for desired in [
        Vector2::new(-4300.0, 1063.1),
        Vector2::new(4300.0, -1063.1),
        Vector2::new(4000.0, 0.0),
        Vector2::new(-4000.0, 900.0),
    ] {
        assert_valid(&sim, sim.valid_free_kick_position(desired));
    }
}
