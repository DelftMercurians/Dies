//! The corner one-timer at real match geometry — simplified for LIVE testing.
//!
//! One fixed play, on a loop: the **striker** kicks the corner ball, the
//! **receiver** posts at the far-side box-edge pocket and redirects it into
//! the goal with the pre-armed reflex kick. All geometry is anchored to a real
//! goal and computed from the live ball, so the play works wherever the ball
//! is actually placed — no assumed seed points (the old fixed ball-relative
//! offsets produced nonsense pockets when the ball wasn't at the sim's
//! corner-analog; live logs 2026-07-04 showed 104° deflections and off-field
//! hints from an own-corner ball).
//!
//! ## Running it live
//!
//! 1. Place the ball in a corner near the tested goal, **at least ~0.5 m off
//!    both lines** — the passer stages ~250 mm behind the ball on the cross
//!    axis and needs in-field room (a ball tight on the touchline wedges the
//!    release against the boundary).
//! 2. The loop idles until the ball is seen settled near a corner of the
//!    tested goal, then: striker frees/approaches the ball → one-timer cross
//!    to the pocket → reflex redirect at the goal centre. After a shot,
//!    place the ball back in a corner (either side — the pocket auto-flips to
//!    the far side).
//!
//! ## Env knobs (set on the dies process; the strategy child inherits them)
//!
//! - `SCENARIO_GOAL=opp` (default) aims at the opponent goal; `own` flips to
//!   our own goal for half-field testing.
//! - `SCENARIO_PASSER` / `SCENARIO_RECEIVER` — robot ids (defaults 1 and 5).

use scenarios::prelude::*;

/// How far from the ball the passer parks before the pass is commanded: past
/// the coordinator's staging point (~200 mm) but well inside its start-distance
/// guard (600 mm).
const FREE_DIST: f64 = 400.0;

/// Lateral standoff of the receive pocket outside the penalty-area side edge.
const POCKET_MARGIN: f64 = 300.0;

/// Ball-to-corner distance within which the play arms (the "ball is placed"
/// gate). Generous — anywhere in the corner region works, the geometry is
/// computed from the actual ball position.
const CORNER_RADIUS: f64 = 2000.0;

fn env_id(name: &str, default: u32) -> PlayerId {
    PlayerId::new(
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default),
    )
}

/// `+1.0` = opponent goal (default), `-1.0` = our own goal (half-field tests).
fn goal_sign() -> f64 {
    match std::env::var("SCENARIO_GOAL").as_deref() {
        Ok("own") => -1.0,
        _ => 1.0,
    }
}

/// Centre of the tested goal.
fn goal_center(world: &World, sgx: f64) -> Vector2 {
    Vector2::new(sgx * world.field_length() / 2.0, 0.0)
}

/// One-timer geometry anchored to the tested goal, from the live ball:
/// pocket = mid-depth point just outside the penalty box's side edge on the
/// side AWAY from the ball (the far post), aim = goal centre. From a corner
/// ball this yields a comfortable ~155–170° deflection; the coordinator's
/// 120° gate still vetoes a genuinely misplaced ball pre-release.
fn corner_geometry(world: &World) -> Option<(Vector2, Vector2)> {
    let ball = world.ball_position()?;
    let sgx = goal_sign();
    let goal = goal_center(world, sgx);
    let pen = if sgx > 0.0 {
        world.opp_penalty_area()
    } else {
        world.own_penalty_area()
    };
    let depth = pen.max.x - pen.min.x;
    let half_w = (pen.max.y - pen.min.y) / 2.0;
    // Far side: opposite the ball's y (deterministic +y for a dead-central ball).
    let side = if ball.y > 0.0 { -1.0 } else { 1.0 };
    let pocket = Vector2::new(
        goal.x - sgx * depth * 0.5,
        side * (half_w + POCKET_MARGIN),
    );
    Some((pocket, goal))
}

/// Idle until the ball is seen settled in the corner region of the tested
/// goal — lets a human walk in and (re)place the ball between legs without the
/// robots thrashing after it.
fn wait_for_corner_ball() -> Step {
    Step::custom("waiting for a corner ball".to_string(), move |ctx| {
        let Some(ball) = ctx.world().ball_position() else {
            return StepOutcome::Running;
        };
        let world = ctx.world();
        let sgx = goal_sign();
        let hx = sgx * world.field_length() / 2.0;
        let hy = world.field_width() / 2.0;
        let settled = world
            .ball_velocity()
            .map(|v| v.norm() < 150.0)
            .unwrap_or(false);
        let near_corner = [hy, -hy]
            .iter()
            .any(|&cy| (ball - Vector2::new(hx, cy)).norm() < CORNER_RADIUS);
        if settled && near_corner {
            // Staging-room check: the passer needs ~250 mm of in-field room
            // behind the ball on the cross axis. A ball tight on a line WILL
            // wedge the release (live logs 2026-07-04: 8 s stage burn, no
            // kick) — warn the humans but still run, so they see it fail live.
            let room = 450.0;
            if world.field_length() / 2.0 - ball.x.abs() < room
                || hy - ball.y.abs() < room
            {
                tracing::warn!(
                    "corner ball only {:.0}/{:.0} mm off the lines — place it ≥{room:.0} mm out or the release will wedge",
                    world.field_length() / 2.0 - ball.x.abs(),
                    hy - ball.y.abs(),
                );
            }
            StepOutcome::Succeeded
        } else {
            StepOutcome::Running
        }
    })
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
        let ball_settled = ctx
            .world()
            .ball_velocity()
            .map(|v| v.norm() < 150.0)
            .unwrap_or(false);
        if !has_ball && ball_settled && dist > FREE_DIST * 0.7 && dist < 550.0 {
            return StepOutcome::Succeeded;
        }
        let dir = (pos - ball)
            .try_normalize(1.0)
            .unwrap_or(Vector2::new(-1.0, 0.0));
        if let Some(h) = ctx.player(passer) {
            h.go_to(ball + dir * FREE_DIST).facing(ball);
        }
        StepOutcome::Running
    })
    .timeout(20.0)
}

fn main() {
    run_scenario(|| {
        let striker = env_id("SCENARIO_PASSER", 1);
        let receiver = env_id("SCENARIO_RECEIVER", 5);
        Scenario::looping(move || {
            vec![
                wait_for_corner_ball(),
                release(striker),
                Step::pass_forward(striker, receiver, |w| corner_geometry(w)).timeout(30.0),
                Step::wait(3.0),
            ]
        })
    });
}
