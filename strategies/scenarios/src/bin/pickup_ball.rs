//! Pick up the ball, then shoot it at the opponent goal — on a loop.
//!
//! Seed a snapshot with the ball near robot 0 before running.

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let r = PlayerId::new(0);
        Scenario::looping(move || {
            vec![
                Step::skill("pickup ball", r, move |h| {
                    h.pickup_ball(Angle::from_radians(0.0));
                })
                .timeout(15.0),
                Step::skill("shoot at goal", r, move |h| {
                    h.reflex_shoot(OPP_GOAL);
                })
                .timeout(10.0),
                Step::wait(0.5),
            ]
        })
    });
}
