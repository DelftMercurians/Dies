use crate::behavior_tree::PassingTarget;
use crate::KickerControlInput::Kick;
use crate::{control::Velocity, PlayerControlInput};
use crate::{find_best_preshoot, ShootTarget};
use dies_core::{Angle, Vector2};
use std::sync::Arc;

use crate::control::{find_nearest_opponent_distance_along_direction, PassingStore};
use crate::skills::{SkillCtx, SkillProgress, SkillResult};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_heading: Option<Angle>,
    state: FetchBallWithPreshootState,
    shoot_target: Option<ShootTarget>,
    set_flag: bool,
    distance_limit: f64,
    avoid_ball_care: f64,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    GoToPreshoot,
    ApproachBall {
        start_pos: Vector2,
        ball_pos: Vector2,
        target_pos: Vector2,
    },
    MoveWithBall {
        target_pos: Vector2,
        start_pos: Vector2,
    },
    Done,
    Failed,
}

impl FetchBallWithPreshoot {
    pub fn new() -> Self {
        Self {
            preshoot_heading: None,
            state: FetchBallWithPreshootState::GoToPreshoot,
            shoot_target: None,
            set_flag: false,
            distance_limit: 160.0,
            avoid_ball_care: 0.0,
        }
    }

    pub fn with_distance_limit(mut self, distance_limit: f64) -> Self {
        self.distance_limit = distance_limit;
        self
    }

    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.preshoot_heading = Some(heading);
        self
    }

    pub fn with_avoid_ball_care(mut self, avoid_ball_care: f64) -> Self {
        self.avoid_ball_care = avoid_ball_care;
        self
    }

    pub fn state(&self) -> String {
        match &self.state {
            FetchBallWithPreshootState::GoToPreshoot => "GoToPreshoot".to_string(),
            FetchBallWithPreshootState::ApproachBall { .. } => "ApproachBall".to_string(),
            FetchBallWithPreshootState::MoveWithBall { .. } => "MoveWithBall".to_string(),
            FetchBallWithPreshootState::Done => "Done".to_string(),
            FetchBallWithPreshootState::Failed => "Failed".to_string(),
        }
    }

    fn should_move_with_ball(&self, ctx: SkillCtx<'_>, target_pos: Vector2) -> bool {
        let player_pos = ctx.player.position;

        // Case 1: We are on our side (neg x) and the shooting target is another robot on the other side (pos x)
        // and our x pos is greater than -900.0
        if let Some(ShootTarget::Player {
            position: Some(target_player_pos),
            ..
        }) = &self.shoot_target
        {
            if player_pos.x < 0.0 && target_player_pos.x > 0.0 && player_pos.x > -900.0 {
                return true;
            }
        }

        // Case 2: Direct line of sight to target is obstructed by a nearby opponent
        let passing_store = PassingStore::new(ctx.player.id, Arc::new(ctx.world.clone()));
        let direction = Angle::between_points(player_pos, target_pos);
        let nearest_opponent_distance =
            find_nearest_opponent_distance_along_direction(&passing_store, direction);

        // Consider obstructed if opponent is within 300mm of the shooting line
        if nearest_opponent_distance < 300.0 {
            return true;
        }

        false
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
                    let shooting_target = find_best_preshoot(
                        &PassingStore::new(ctx.player.id, Arc::new(ctx.world.clone())),
                        self.shoot_target.clone(),
                    );
                    self.shoot_target = Some(shooting_target.clone());
                    let shooting_target = shooting_target.position().unwrap();
                    let prep_target = ball_pos - (shooting_target - ball_pos).normalize() * 180.0;

                    let distance_to_prep_target = (prep_target - player_pos).magnitude();
                    if distance_to_prep_target < 15.0
                        || (distance_to_prep_target < 80.0 && ctx.player.velocity.norm() < 30.0)
                    {
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
                    input.care = self.avoid_ball_care;

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
                        // Check if we should move with ball before shooting
                        if self.should_move_with_ball(ctx, *target_pos) {
                            self.state = FetchBallWithPreshootState::MoveWithBall {
                                target_pos: *target_pos,
                                start_pos: player_pos,
                            };
                            return SkillProgress::Continue(input);
                        } else {
                            input.with_kicker(Kick { force: 1.0 });
                            self.state = FetchBallWithPreshootState::Done;
                            return SkillProgress::Continue(input);
                        }
                    }

                    if (player_pos - start_pos).magnitude() > self.distance_limit {
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Failed;
                    }

                    // Move forward towards the ball
                    input.velocity = Velocity::global(ball_heading.to_vector() * 300.0);

                    SkillProgress::Continue(input)
                }
                FetchBallWithPreshootState::MoveWithBall {
                    target_pos,
                    start_pos,
                } => {
                    // Higher dribbling speed for ball control
                    input.with_dribbling(1.0);

                    // Maintain heading toward target with lowered yaw rate
                    let target_heading = Angle::between_points(player_pos, *target_pos);
                    input.with_yaw(target_heading);
                    input.with_angular_speed_limit(1.5);

                    // Check distance to target - kick when close enough
                    let distance_to_target = (player_pos - *target_pos).magnitude();
                    if distance_to_target < 50.0 {
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                    }

                    // Check if we've moved too far
                    let distance_moved = (player_pos - *start_pos).magnitude();
                    if distance_moved > 900.0 {
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Failed;
                        return SkillProgress::Continue(input);
                    }

                    input.with_position(*target_pos);
                    input.with_speed_limit(500.0);

                    SkillProgress::Continue(input)
                }
                FetchBallWithPreshootState::Done => SkillProgress::Done(SkillResult::Success),
                FetchBallWithPreshootState::Failed => SkillProgress::Done(SkillResult::Failure),
            }
        } else {
            // Wait for the ball to appear
            SkillProgress::Continue(PlayerControlInput::default())
        }
    }
}
