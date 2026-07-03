//! Re-aim from possession — on a loop. Exercises the `"HandleBall re-aim"`
//! flag: capture the ball facing AWAY from the target, then command a
//! drive-through `Strike` at it. The fresh skill starts from possession ~180°
//! off the kick axis — exactly the post-receive continuation shape — and must
//! re-aim per the flag: disengage-and-re-strike (`REAIM_DISENGAGE` > 0.5,
//! default) or orbit with the reflex fired once aligned (≤ 0.5).
//!
//! Watch `hb.stage` (`disengage` → `strike` vs `aim`), `hb.reaim`, and the
//! `hb.arm_guard` counter (must stay 0 — a nonzero value means the routing let
//! a held, misaligned ball reach the commit latch).
//!
//! Seed a snapshot with the ball near robot 3 before running (e.g.
//! `pass_settle_verify`).

use std::f64::consts::PI;

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let r = PlayerId::new(3);
        Scenario::looping(move || {
            vec![
                // Capture + hold facing our own goal (−x): the follow-up kick
                // toward +x is the maximal re-aim.
                Step::skill("acquire + hold facing away", r, move |h| {
                    h.handle_ball(
                        BallAction::Hold {
                            heading: Angle::from_radians(PI),
                        },
                        AcquirePosition::Heading(Angle::from_radians(PI)),
                    );
                })
                .timeout(10.0),
                // Fresh drive-through Strike from possession: the skill must
                // re-aim (disengage or orbit) and release at the target.
                Step::skill("re-aim strike at goal", r, move |h| {
                    h.handle_ball(
                        BallAction::Strike {
                            target: Vector2::new(4000.0, 0.0),
                            acquire_first: false,
                        },
                        AcquirePosition::Default,
                    );
                })
                .timeout(15.0),
                Step::wait(3.0),
            ]
        })
    });
}
