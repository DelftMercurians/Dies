//! Exercise the unified `HandleBall` skill — on a loop.
//!
//! Acquires the ball and holds it facing *away* from the opponent goal, then
//! live-swaps the action to `Shoot`: because both steps issue `HandleBall`, the
//! executor updates the running skill's params in place, so the held ball is
//! aimed (orbited onto the shot axis) and kicked without tearing the skill down —
//! the acquire→hold→shoot seam the merged skill exists to remove.
//!
//! Seed a snapshot with the ball near robot 0 before running.

use std::f64::consts::PI;

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let r = PlayerId::new(0);
        Scenario::looping(move || {
            vec![
                // Acquire + hold, facing -x (away from goal) so the follow-up
                // Shoot has to orbit the ball to aim.
                Step::skill("acquire + hold", r, move |h| {
                    h.handle_ball(
                        BallAction::Hold {
                            heading: Angle::from_radians(PI),
                        },
                        None,
                    );
                })
                .timeout(15.0),
                // Live-swap to Shoot on the same skill instance: aim + kick.
                Step::skill("shoot at goal", r, move |h| {
                    h.handle_ball(BallAction::Shoot { target: OPP_GOAL }, None);
                })
                .timeout(15.0),
                Step::wait(0.5),
            ]
        })
    });
}
