//! Hold a move command while the game is stopped, then fire the kick command
//! once when play runs.

use scenarios::prelude::*;

fn main() {
    run_scenario(|| {
        let robot = PlayerId::new(5);
        let move_target = Vector2::new(-2500.0, 1000.0);
        let junior_target = Vector2::new(-500.0, -3000.0);
        let kick_target = Vector2::new(0.0, -500.0);
        let mut ran_second_skill_this_run = false;

        Scenario::once(move || {
            vec![Step::custom("move on stop, kick on run", move |ctx| {
                let game_state = ctx.world().game_state();

                let Some(player) = ctx.player(robot) else {
                    return StepOutcome::Running;
                };

                if game_state == GameState::Run {
                    if !ran_second_skill_this_run {
                        println!("Game state is Run, executing kick skill");
                        player.strike_through(kick_target);
                        player.set_role("KickToPoint");
                        ran_second_skill_this_run = true;
                    }
                } else {
                    ran_second_skill_this_run = false;
                    player.go_to(move_target);
                    player.set_role("MoveToBall");
                }

                StepOutcome::Running
            })]
        })
    });
}