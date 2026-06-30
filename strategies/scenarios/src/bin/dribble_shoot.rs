//! Capture the ball facing away from goal, then orbit-and-shoot it toward the
//! opponent goal with the unified dribble-shoot skill — on a loop.
//!
//! Seed a snapshot with the ball near robot 0 before running.

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let r = PlayerId::new(0);
        Scenario::looping(move || {
            vec![
                // Capture facing -x (away from the opponent goal) so the
                // dribble-shoot has to orbit the ball to aim.
                Step::skill("pickup ball", r, move |h| {
                    h.pickup_ball(Angle::from_radians(std::f64::consts::PI));
                })
                .timeout(15.0),
                Step::skill("dribble-shoot at goal", r, move |h| {
                    h.dribble_shoot(OPP_GOAL);
                })
                .timeout(15.0),
                Step::wait(0.5),
            ]
        })
    });
}
