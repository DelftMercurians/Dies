//! One-timer passes on a loop: pass to a teammate that redirects the ball
//! toward the goal with the pre-armed reflex kick — the pass→shot chain with
//! zero handling latency. Exercises `ctx.pass(..).forward_to(..)`, the
//! coordinator's one-timer mode, and the `Success { forwarded }` verdict.
//!
//! ## Geometry
//!
//! The receiver faces the FORWARD target, so the incoming pass reaches its
//! kicker face only when `angle(ball→hint, hint→goal)` is large — 180° is a
//! head-on return, 90° the physical limit, and the coordinator rejects
//! anything under its `ONE_TIMER_MIN_DEFLECTION` bar (`InfeasibleGeometry`).
//! The receive hint is therefore computed per loop from wherever the ball
//! settled: backed off the ball→goal line by [`HINT_BACKOFF`] with a
//! [`HINT_LATERAL`] offset (alternating side per leg), which yields a ~165°
//! head-on deflection and keeps the return shot clear of the parked passer.
//!
//! ## Which goal
//!
//! `SCENARIO_GOAL=opp` (default) shoots at the opponent goal; `SCENARIO_GOAL=own`
//! flips to our own goal for half-field testing. Set it on the dies process
//! (the strategy child inherits it): `SCENARIO_GOAL=own cargo run -- --strategy
//! pass_forward`.
//!
//! Self-seeding works like the catch-pass scenario (`pass.rs`): the pass never
//! holds and never chases, so between passes the next passer first *frees* the
//! ball by backing off to a comfortable staging distance.

use scenarios::prelude::*;

/// How far from the ball the passer parks before the pass is commanded: past
/// the coordinator's staging point (~200 mm) but well inside its start-distance
/// guard (600 mm).
const FREE_DIST: f64 = 400.0;

/// How far the receive hint backs off the ball along the ball→goal line. The
/// pass travels away from the goal into the receiver's mouth; the reflex fires
/// it back past the ball toward the goal.
const HINT_BACKOFF: f64 = 2200.0;

/// Lateral offset of the receive hint off the ball→goal line, so the return
/// shot doesn't go straight back through the passer. With [`HINT_BACKOFF`]
/// this keeps the deflection ~165° (head-on) and clears the passer by ~0.7 m.
const HINT_LATERAL: f64 = 1000.0;

/// One-timer geometry from the live ball: hint backed off the ball→goal line
/// on side `side` (±1), forwarding to `goal`. `None` until the ball is seen.
fn one_timer_geometry(world: &World, goal: Vector2, side: f64) -> Option<(Vector2, Vector2)> {
    let ball = world.ball_position()?;
    let u = (goal - ball).try_normalize(1.0)?;
    let perp = Vector2::new(-u.y, u.x);
    let hint = ball - u * HINT_BACKOFF + perp * (side * HINT_LATERAL);
    Some((hint, goal))
}

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
    let goal = match std::env::var("SCENARIO_GOAL").as_deref() {
        Ok("own") => OWN_GOAL,
        _ => OPP_GOAL,
    };
    run_scenario(move || {
        let a = PlayerId::new(4);
        let b = PlayerId::new(0);
        Scenario::looping(move || {
            vec![
                release(a),
                Step::pass_forward(a, b, move |w| one_timer_geometry(w, goal, 1.0))
                    .timeout(30.0),
                Step::wait(3.0),
                release(b),
                Step::pass_forward(b, a, move |w| one_timer_geometry(w, goal, -1.0))
                    .timeout(30.0),
                Step::wait(3.0),
            ]
        })
    });
}
