//! Bounce the ball between two robots with the joint pass coordinator — on a
//! loop. Exercises `ctx.pass`, the typed `PassResult`, and the drive-through
//! reflex-strike release.
//!
//! Self-seeding: the pass never holds the ball and never chases a loose one —
//! it fails fast if the passer already possesses the ball or if the ball is too
//! far away. So between passes the next passer first *frees* the ball: after a
//! catch the ball sits on the receiver's dribbler, and backing away from it IS
//! the release (no skill ever "hands over" a held ball). The `release` step
//! backs the robot off to a comfortable staging distance (still well inside the
//! pass's start-distance guard), then the pass stages and strikes.

use scenarios::prelude::*;

/// How far from the ball the passer parks before the pass is commanded: past
/// the coordinator's staging point (~200 mm) but well inside its start-distance
/// guard (600 mm).
const FREE_DIST: f64 = 400.0;

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
        if !has_ball && dist > FREE_DIST * 0.7 && dist < 550.0 {
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
                Step::pass(a, b, None).timeout(15.0),
                Step::wait(10.0),
                release(b),
                Step::pass(b, a, None).timeout(15.0),
                Step::wait(10.0),
            ]
        })
    });
}
