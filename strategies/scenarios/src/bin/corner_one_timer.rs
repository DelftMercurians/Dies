//! The corner one-timer at its **real match geometry** — on a loop.
//!
//! Concerto's corner play (see `strategies/concerto`, `CORNER_OT_*`) crosses
//! from a corner carrier to a pocket at the opponent box edge, which redirects
//! the ball into the goal with the pre-armed reflex kick. Its deflection is
//! tight — ~135° against the coordinator's 120° `ONE_TIMER_MIN_DEFLECTION`
//! gate — unlike the `pass_forward` scenario's comfortable ~165° head-on
//! return. This scenario exercises the coordinator at exactly that marginal
//! geometry: cross ≈ 3.5 m, deflection ≈ 136°.
//!
//! ## Why the positions look shifted
//!
//! The whole figure (ball → pocket → aim) is the real corner play translated
//! 2.5 m away from the opponent goal, into open field. Deflection angles are
//! translation-invariant, and the redirect must NOT fire into the physical
//! goal — a scored ball wedges in the net and kills the loop (the same reason
//! `pass_forward` aims at a virtual point). Legs alternate the mirrored ±y
//! "corner" so the field re-seeds itself.
//!
//! Loop: carrier dribbles the ball to the corner-analog point, frees it
//! (the pass never starts on a held ball), then the one-timer runs with the
//! pocket/aim placed relative to wherever the ball actually settled.

use scenarios::prelude::*;

/// Ball seed point of the +y leg — the corner-analog (real corner ≈
/// (4100, 2450), translated −2500 in x away from the goal). Kept ≥550 mm off
/// the touchline: the passer stages ~250 mm BEHIND the ball on the cross axis,
/// and the carry can settle the ball up to 250 mm past the seed — a deeper
/// seed wedges the release against the boundary (the exact failure the
/// planner's `CORNER_OT_STAGE_ROOM` gate excludes in the real play).
const SEED: Vector2 = Vector2::new(1600.0, 2450.0);

/// Pocket relative to the settled ball (real play: ball (4100, 2600) →
/// box-edge pocket (3200, −800)). Yields ≈136° deflection with [`AIM_REL`].
const POCKET_REL: Vector2 = Vector2::new(-900.0, -3400.0);

/// Redirect aim relative to the settled ball (real play: goal centre
/// (4500, 0)).
const AIM_REL: Vector2 = Vector2::new(400.0, -2600.0);

/// How far from the ball the passer parks before the pass is commanded: past
/// the coordinator's staging point (~200 mm) but well inside its start-distance
/// guard (600 mm).
const FREE_DIST: f64 = 400.0;

/// Margin (mm) kept between the receive pocket and the field boundary.
const FIELD_MARGIN: f64 = 190.0;

/// Clamp `pos` into the field playing area (minus [`FIELD_MARGIN`]).
fn clamp_to_field(world: &World, pos: Vector2) -> Vector2 {
    let hx = world.field_length() / 2.0 - FIELD_MARGIN;
    let hy = world.field_width() / 2.0 - FIELD_MARGIN;
    Vector2::new(pos.x.clamp(-hx, hx), pos.y.clamp(-hy, hy))
}

/// One-timer geometry from the live ball: pocket and aim at the corner play's
/// relative offsets, mirrored per leg by `side` (±1). `None` until the ball is
/// seen.
fn corner_geometry(world: &World, side: f64) -> Option<(Vector2, Vector2)> {
    let ball = world.ball_position()?;
    let mirror = |v: Vector2| Vector2::new(v.x, v.y * side);
    let pocket = clamp_to_field(world, ball + mirror(POCKET_REL));
    let aim = ball + mirror(AIM_REL);
    Some((pocket, aim))
}

/// Dribble the ball to `dest` (the leg's corner-analog), facing the upcoming
/// pocket so the release backs straight off the cross axis. Succeeds once the
/// ball has settled at the destination.
fn carry_to(r: PlayerId, dest: Vector2, side: f64) -> Step {
    Step::custom(format!("{r} carries the ball to the corner"), move |ctx| {
        let Some(ball) = ctx.world().ball_position() else {
            return StepOutcome::Running;
        };
        let settled = ctx
            .world()
            .ball_velocity()
            .map(|v| v.norm() < 150.0)
            .unwrap_or(false);
        if (ball - dest).norm() < 250.0 && settled {
            return StepOutcome::Succeeded;
        }
        if let Some(h) = ctx.player(r) {
            let pocket_dir = Vector2::new(POCKET_REL.x, POCKET_REL.y * side);
            h.handle_ball(
                BallAction::Carry {
                    to: dest,
                    heading: Angle::from_radians(pocket_dir.y.atan2(pocket_dir.x)),
                },
                AcquirePosition::Default,
            );
        }
        StepOutcome::Running
    })
    .timeout(40.0)
}

/// Park the upcoming receiver far from the play — every leg is a COLD
/// cross-field pull, like the real corner play (the receiver is nominated from
/// wherever it happens to be, often the other side of the field). Without this
/// the loop settles into a rhythm where the receiver starts each leg already
/// posted near the pocket, which hides arrival-transient misses (observed:
/// warm legs 100%, cold legs missed).
fn pull_away(r: PlayerId, to: Vector2) -> Step {
    Step::skill(format!("{r} parks far (cold pull)"), r, move |h| {
        h.go_to(to);
    })
    .timeout(15.0)
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
    .timeout(15.0)
}

fn main() {
    run_scenario(|| {
        let a = PlayerId::new(3);
        let b = PlayerId::new(4);
        Scenario::looping(move || {
            vec![
                // +y corner: a crosses, b is pulled COLD from across the field
                // to the pocket and redirects.
                carry_to(a, SEED, 1.0),
                pull_away(b, Vector2::new(-1200.0, -1800.0)),
                release(a),
                Step::pass_forward(a, b, |w| corner_geometry(w, 1.0)).timeout(30.0),
                Step::wait(3.0),
                // Mirrored −y corner: roles swap, the field re-seeds itself.
                carry_to(b, Vector2::new(SEED.x, -SEED.y), -1.0),
                pull_away(a, Vector2::new(-1200.0, 1800.0)),
                release(b),
                Step::pass_forward(b, a, |w| corner_geometry(w, -1.0)).timeout(30.0),
                Step::wait(3.0),
            ]
        })
    });
}
