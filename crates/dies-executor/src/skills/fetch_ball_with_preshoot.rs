use crate::KickerControlInput::Kick;
use crate::{control::Velocity, PlayerControlInput};
use dies_core::{Angle, Vector2};
use std::sync::Arc;

use crate::control::{find_best_preshoot_target_target, PassingStore};
use crate::skills::{SkillCtx, SkillProgress, SkillResult};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_heading: Option<Angle>,
    state: FetchBallWithPreshootState,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    GoToPreshoot,
    ApproachBall {
        start_pos: Vector2,
        ball_pos: Vector2,
    },
    Done,
}

impl FetchBallWithPreshoot {
    pub fn new() -> Self {
        Self {
            preshoot_heading: None,
            state: FetchBallWithPreshootState::GoToPreshoot,
        }
    }

    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.preshoot_heading = Some(heading);
        self
    }

    pub fn state(&self) -> String {
        match &self.state {
            FetchBallWithPreshootState::GoToPreshoot => "GoToPreshoot".to_string(),
            FetchBallWithPreshootState::ApproachBall { .. } => "ApproachBall".to_string(),
            FetchBallWithPreshootState::Done => "Done".to_string(),
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        // go to preshoot:
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            input.with_dribbling(0.6);

            let ball_pos = ball.position.xy();
            let ball_speed = ball.velocity.xy().norm();
            let player_pos = ctx.player.position;

            match &self.state {
                FetchBallWithPreshootState::GoToPreshoot => {
                    let shooting_target = find_best_preshoot_target_target(&PassingStore::new(
                        ctx.player.id,
                        Arc::new(ctx.world.clone()),
                    ))
                    .position()
                    .unwrap();
                    let prep_target = ball_pos - (shooting_target - ball_pos).normalize() * 180.0;

                    let distance_to_prep_target = (prep_target - player_pos).magnitude();
                    if distance_to_prep_target < 30.0 {
                        self.state = FetchBallWithPreshootState::ApproachBall {
                            start_pos: player_pos,
                            ball_pos,
                        };
                        return SkillProgress::Continue(input);
                    }

                    input.with_position(prep_target);
                    let ball_heading = Angle::between_points(prep_target, ball_pos);
                    input.with_yaw(ball_heading);
                    input.avoid_ball = true;

                    SkillProgress::Continue(input)
                }
                FetchBallWithPreshootState::ApproachBall {
                    start_pos,
                    ball_pos,
                } => {
                    if ctx.player.breakbeam_ball_detected {
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                    }

                    if (player_pos - start_pos).magnitude() > 160.0 {
                        return SkillProgress::Done(SkillResult::Failure);
                    }

                    let ball_heading = Angle::between_points(player_pos, *ball_pos);
                    input.with_yaw(ball_heading);
                    input.avoid_ball = true;

                    // Move forward towards the ball
                    input.velocity = Velocity::global(ball_heading.to_vector() * 300.0);

                    SkillProgress::Continue(input)
                }
                FetchBallWithPreshootState::Done => SkillProgress::Done(SkillResult::Success),
            }
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}
