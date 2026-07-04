//! Move robot 0 to `(-400, -400)`, then capture the ball and kick it toward
//! `(-2000, -2000)`.

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let robot = PlayerId::new(5);
        let move_target = Vector2::new(-2500.0, 1000.0);
        let junior_target = Vector2::new(-500.0, -3000.0);
        let kick_target = Vector2::new(300.0, -2000.0);

        Scenario::once(move || {
            vec![
                Step::skill("move to position", robot, move |player| {
                    player.go_to(move_target);
                    player.set_role("MoveToBall");

                })
                .timeout(15.0),
                Step::skill("capture and kick", robot, move |player| {
                    player.strike_through(kick_target);
                    player.set_role("KickToPoint");
                })
                .timeout(20.0),
            ]
        })
    });
}