//! Bounce the ball between two robots with the joint pass coordinator — on a
//! loop. Exercises `ctx.pass`, the typed `PassResult`, and (in the secure step)
//! the unified `handle_ball` capture.
//!
//! Self-seeding: the pass coordinator never *chases* a loose ball — its Secure
//! phase fails instantly with `BallLost` if the passer isn't already on the ball.
//! So before each pass the next passer first drives onto the ball and captures it
//! via `handle_ball`, completing on possession. This means the scenario works
//! from a default field too — no snapshot required — though seeding the ball near
//! robot 0 still makes the first capture quicker.

use scenarios::prelude::*;

/// Drive `passer` onto the ball and hold it (facing `receiver`), completing once
/// it has possession. Guarantees the following [`Step::pass`] has a ball to
/// secure instead of failing `BallLost` on a loose ball.
///
/// `handle_ball` is **one-shot**: its acquire backstop latches `Failed` if the
/// robot is kept off the ball long enough (e.g. while the auto-ref holds robots
/// ~1 m off the ball during a kickoff), and a re-issued same-type command will
/// *not* restart a latched one-shot. So on `Failed` we bounce through `stop` for
/// one frame to clear the latch, then retry with a fresh instance — the capture
/// recovers as soon as the robot is allowed to reach the ball.
fn secure(passer: PlayerId, receiver: PlayerId) -> Step {
    Step::custom(format!("{passer} secures ball"), move |ctx| {
        if ctx
            .player_ref(passer)
            .map(|p| p.has_ball())
            .unwrap_or(false)
        {
            return StepOutcome::Succeeded;
        }
        // Face the receiver at capture so the pass can align with less re-orbit.
        let heading = match (ctx.world().ball_position(), ctx.player_ref(receiver)) {
            (Some(ball), Some(rx)) => Angle::between_points(ball, rx.position()),
            _ => Angle::from_radians(0.0),
        };
        let latched = ctx
            .player_ref(passer)
            .map(|p| p.skill_status() == SkillStatus::Failed)
            .unwrap_or(false);
        if let Some(p) = ctx.player(passer) {
            if latched {
                p.stop(); // clear the one-shot latch; retry next frame
            } else {
                p.handle_ball(BallAction::Hold { heading }, AcquirePosition::Default);
            }
        }
        StepOutcome::Running
    })
    .timeout(30.0)
}

fn main() {
    run_scenario(|| {
        let a = PlayerId::new(0);
        let b = PlayerId::new(1);
        Scenario::looping(move || {
            vec![
                secure(a, b),
                Step::pass(a, b, None).timeout(20.0),
                Step::wait(0.5),
                secure(b, a),
                Step::pass(b, a, None).timeout(20.0),
                Step::wait(0.5),
            ]
        })
    });
}
