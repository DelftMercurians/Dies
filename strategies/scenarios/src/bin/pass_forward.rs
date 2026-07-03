//! One-timer passes on a loop: pass to a teammate that redirects the ball
//! toward the goal with the pre-armed reflex kick — the pass→shot chain with
//! zero handling latency. Exercises `ctx.pass(..).forward_to(..)`, the
//! coordinator's one-timer mode, and the `Success { forwarded }` verdict.
//!
//! Geometry: the receive point is hinted off the ball→goal line so the
//! deflection angle (incoming pass vs receiver→goal) stays shallow (~45°) —
//! the coordinator trusts the caller here and only emits
//! `pass.<a>-<b>.deflection_deg` for observability. After each shot the ball
//! usually ends up out of play / in the goal and the auto-ref re-centers it,
//! so each loop iteration starts from a similar layout.
//!
//! Self-seeding works like the catch-pass scenario (`pass.rs`): the pass never
//! holds and never chases, so between passes the next passer first *frees* the
//! ball by backing off to a comfortable staging distance.

use scenarios::prelude::*;

/// How far from the ball the passer parks before the pass is commanded: past
/// the coordinator's staging point (~200 mm) but well inside its start-distance
/// guard (600 mm).
const FREE_DIST: f64 = 400.0;

/// Receive points for the one-timer, offset laterally from the center→goal
/// line: from a ball near midfield the incoming pass and the outgoing
/// redirect toward [`OPP_GOAL`] meet at a shallow (~45°) deflection.
const HINT_HIGH: Vector2 = Vector2::new(-2500.0, 900.0);
const HINT_LOW: Vector2 = Vector2::new(-2500.0, -900.0);

/// Free the ball for `passer`: if the robot holds it (or is pressed against
/// it), back straight away from the ball to [`FREE_DIST`]; if it is too far,
/// walk in. Succeeds once the ball is free and at a passable distance.
fn release(passer: PlayerId) -> Step {
    Step::custom(format!("{passer} frees the ball"), move |ctx| {
        let (Some(ball), Some(p)) = (ctx.world().ball_position(), ctx.player_ref(passer)) else {
            return StepOutcome::Running;
        };
        let pos = p.position();
        let has_ball = p.has_ball();
        let dist = (pos - ball).norm();
        // Only hand over to the pass once the ball has settled — a pass issued
        // on a still-rolling ball (e.g. right after the previous deflection)
        // trips its ball-moved abort within a few frames.
        let ball_settled = ctx
            .world()
            .ball_velocity()
            .map(|v| v.norm() < 150.0)
            .unwrap_or(false);
        if !has_ball && ball_settled && dist > FREE_DIST * 0.7 && dist < 550.0 {
            return StepOutcome::Succeeded;
        }
        // Back away from (or walk toward) a point FREE_DIST off the ball, on
        // the robot's side of it, facing the ball the whole time.
        let dir = (pos - ball)
            .try_normalize(1.0)
            .unwrap_or(Vector2::new(-1.0, 0.0));
        if let Some(h) = ctx.player(passer) {
            h.go_to(ball + dir * FREE_DIST).facing(ball);
        }
        StepOutcome::Running
    })
    .timeout(15.0)
}

fn main() {
    run_scenario(|| {
        let a = PlayerId::new(4);
        let b = PlayerId::new(0);
        Scenario::looping(move || {
            vec![
                release(a),
                Step::pass_forward(a, b, Some(HINT_HIGH), Vector2::new(-4500.0, 0.0)).timeout(30.0),
                Step::wait(3.0),
                release(b),
                Step::pass_forward(b, a, Some(HINT_LOW), Vector2::new(-4500.0, 0.0)).timeout(30.0),
                Step::wait(3.0),
            ]
        })
    });
}
