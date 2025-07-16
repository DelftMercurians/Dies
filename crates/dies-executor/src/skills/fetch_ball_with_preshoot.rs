use crate::behavior_tree::PassingTarget;
use crate::KickerControlInput::Kick;
use crate::ShootTarget;
use crate::{control::Velocity, PlayerControlInput};
use dies_core::{Angle, Vector2};
use std::sync::Arc;

use crate::control::{find_best_preshoot_target_target, PassingStore};
use crate::skills::{SkillCtx, SkillProgress, SkillResult};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_heading: Option<Angle>,
    state: FetchBallWithPreshootState,
    shoot_target: Option<ShootTarget>,
    set_flag: bool,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    GoToPreshoot,
    ApproachBall {
        start_pos: Vector2,
        ball_pos: Vector2,
        target_pos: Vector2,
    },
    Done,
}

impl FetchBallWithPreshoot {
    pub fn new() -> Self {
        Self {
            preshoot_heading: None,
            state: FetchBallWithPreshootState::GoToPreshoot,
            shoot_target: None,
            set_flag: false,
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
                    ));
                    self.shoot_target = Some(shooting_target.clone());
                    let shooting_target = shooting_target.position().unwrap();
                    let prep_target = ball_pos - (shooting_target - ball_pos).normalize() * 180.0;

                    let distance_to_prep_target = (prep_target - player_pos).magnitude();
                    if distance_to_prep_target < 10.0 {
                        self.state = FetchBallWithPreshootState::ApproachBall {
                            start_pos: player_pos,
                            ball_pos,
                            target_pos: shooting_target,
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
                    target_pos,
                } => {
                    if let Some(ShootTarget::Player { id, position }) = &self.shoot_target {
                        if !self.set_flag {
                            self.set_flag = true;
                            ctx.bt_context.set_passing_target(PassingTarget {
                                id: *id,
                                position: position.unwrap_or(ctx.world.get_player(*id).position),
                            });
                        }
                    }

                    let ball_heading = Angle::between_points(player_pos, *target_pos);
                    input.with_yaw(ball_heading);
                    input.avoid_ball = true;

                    if ctx.player.breakbeam_ball_detected {
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                    }

                    if (player_pos - start_pos).magnitude() > 160.0 {
                        return SkillProgress::Done(SkillResult::Failure);
                    }

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
