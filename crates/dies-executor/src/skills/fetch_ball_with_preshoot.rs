use crate::behavior_tree::PassingTarget;
use crate::KickerControlInput::Kick;
use crate::{control::Velocity, PlayerControlInput};
use crate::{find_best_preshoot, ShootTarget};
use dies_core::{Angle, Vector2, PLAYER_RADIUS};
use std::fmt::format;
use std::sync::Arc;
use std::time::Instant;

use crate::control::PassingStore;
use crate::skills::{SkillCtx, SkillProgress, SkillResult};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_heading: Option<Angle>,
    state: FetchBallWithPreshootState,
    shoot_target: Option<ShootTarget>,
    can_pass: bool,
    distance_limit: f64,
    avoid_ball_care: f64,
    go_to_preshoot_timer: Option<Instant>,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    GoToPreshoot {
        start_time: Option<Instant>,
    },
    ApproachBall {
        start_pos: Vector2,
        ball_pos: Vector2,
        target_pos: Vector2,
    },
    MoveWithBall {
        target_pos: Vector2,
        start_pos: Vector2,
        go_to_pos: Vector2,
    },
    Done,
    Failed,
}

impl FetchBallWithPreshoot {
    pub fn new() -> Self {
        Self {
            preshoot_heading: None,
            state: FetchBallWithPreshootState::GoToPreshoot { start_time: None },
            shoot_target: None,
            can_pass: true,
            distance_limit: 160.0,
            avoid_ball_care: 0.0,
            go_to_preshoot_timer: None,
        }
    }

    pub fn with_can_pass(mut self, can_pass: bool) -> Self {
        self.can_pass = can_pass;
        self
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
            FetchBallWithPreshootState::GoToPreshoot { .. } => "GoToPreshoot".to_string(),
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
        if let Some(ShootTarget::Goal(_)) = &self.shoot_target {
            if player_pos.x < 0.0 && player_pos.x > -900.0 {
                return true;
            }
        }

        // // Case 2: Direct line of sight to target is obstructed by a nearby opponent
        // let passing_store = PassingStore::new(ctx.player.id, Arc::new(ctx.world.clone()));
        // let direction = Angle::between_points(player_pos, target_pos);
        // let nearest_opponent_distance =
        //     find_nearest_opponent_distance_along_direction(&passing_store, direction);

        // // Consider obstructed if opponent is within 300mm of the shooting line
        // if nearest_opponent_distance < 300.0 {
        //     return true;
        // }

        false
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let mut input = PlayerControlInput::new();
            input.with_dribbling(0.6);

            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position;
            let player_heading = ctx.player.yaw;

            let dribbler_position = player_pos + player_heading.to_vector() * PLAYER_RADIUS;
            let dribbler_radius = if ball_pos.norm() < 1500.0 { 30.0 } else { 15.0 };

            match &self.state {
                FetchBallWithPreshootState::GoToPreshoot { .. } => {
                    let start_time = self.go_to_preshoot_timer.get_or_insert(Instant::now());
                    let shooting_target = find_best_preshoot(
                        &PassingStore::new(ctx.player.id, Arc::new(ctx.world.clone())),
                        self.shoot_target.clone(),
                        self.can_pass,
                    );
                    if let Some(ShootTarget::Player { id, position }) = &self.shoot_target {
                        ctx.bt_context.set_passing_target(PassingTarget {
                            shooter_id: ctx.player.id,
                            id: *id,
                            position: position.unwrap_or(ctx.world.get_player(*id).position),
                        });
                    }
                    ctx.team_context.debug_cross_colored(
                        "shooting_target",
                        shooting_target.position().unwrap_or_default(),
                        dies_core::DebugColor::Blue,
                    );
                    self.shoot_target = Some(shooting_target.clone());
                    let shooting_target = shooting_target.position().unwrap();
                    let prep_target = ball_pos - (shooting_target - ball_pos).normalize() * 150.0;

                    let ball_to_dribbler_distance = (ball_pos - dribbler_position).magnitude();
                    let distance_to_prep_target = (prep_target - player_pos).magnitude();
                    if distance_to_prep_target < 15.0
                        || (distance_to_prep_target < 90.0
                            && ctx.player.velocity.norm() < 10.0
                            && start_time.elapsed().as_secs_f64() > 2.0)
                        || ball_to_dribbler_distance < dribbler_radius
                    {
                        self.state = FetchBallWithPreshootState::ApproachBall {
                            start_pos: player_pos,
                            ball_pos,
                            target_pos: shooting_target,
                        };
                        return SkillProgress::Continue(input);
                    }
                    if start_time.elapsed().as_secs_f64() > 5.0 {
                        return SkillProgress::Done(SkillResult::Failure);
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
                    let start_time = self.go_to_preshoot_timer.get_or_insert(Instant::now());
                    if let Some(ShootTarget::Player { id, position }) = &self.shoot_target {
                        ctx.bt_context.set_passing_target(PassingTarget {
                            shooter_id: ctx.player.id,
                            id: *id,
                            position: position.unwrap_or(ctx.world.get_player(*id).position),
                        });
                    }

                    let target_heading = Angle::between_points(player_pos, *target_pos);
                    input.with_yaw(target_heading);
                    input.avoid_ball = true;

                    let ball_heading = Angle::between_points(player_pos, *ball_pos);
                    let ball_distance = (ball_pos - player_pos).magnitude();
                    let ball_to_dribbler_distance = (ball_pos - dribbler_position).magnitude();
                    dies_core::debug_value("ball_to_dribbler_distance", ball_to_dribbler_distance);
                    if ball_to_dribbler_distance < dribbler_radius {
                        // Check if we should move with ball before shooting
                        // if self.should_move_with_ball(ctx, *target_pos) {
                        // println!("Move with ball: target_pos={:.2}", target_pos);
                        // self.state = FetchBallWithPreshootState::MoveWithBall {
                        //     target_pos: *target_pos,
                        //     start_pos: player_pos,
                        //     go_to_pos: Vector2::new(100.0, player_pos.y),
                        // };
                        // return SkillProgress::Continue(input);
                        // } else {
                        //     println!(
                        //         "Fetch ball done: ball_to_dribbler_distance<{:.2}",
                        //         ball_to_dribbler_distance
                        //     );
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                        // }
                    }

                    if (player_pos - start_pos).magnitude() > self.distance_limit {
                        println!(
                            "Failed to fetch ball: distance_limit<{:.2}",
                            (player_pos - start_pos).magnitude()
                        );
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                    }

                    if ball_distance > 500.0 || start_time.elapsed().as_secs_f64() > 5.0 {
                        println!(
                            "Failed to fetch ball: ball_distance={:.2}, time={:.2}",
                            ball_distance,
                            start_time.elapsed().as_secs_f64()
                        );
                        return SkillProgress::Done(SkillResult::Failure);
                    }

                    // Move forward towards the ball
                    input.velocity = Velocity::global(ball_heading.to_vector() * 400.0);

                    SkillProgress::Continue(input)
                }
                FetchBallWithPreshootState::MoveWithBall {
                    target_pos,
                    start_pos,
                    go_to_pos,
                } => {
                    if let Some(ShootTarget::Player { id, position }) = &self.shoot_target {
                        ctx.bt_context.set_passing_target(PassingTarget {
                            shooter_id: ctx.player.id,
                            id: *id,
                            position: position.unwrap_or(ctx.world.get_player(*id).position),
                        });
                    }

                    let ball_to_dribbler_distance = (ball_pos - dribbler_position).magnitude();
                    dies_core::debug_value("ball_to_dribbler_distance", ball_to_dribbler_distance);

                    // Higher dribbling speed for ball control
                    input.with_dribbling(1.0);

                    // Maintain heading toward target with lowered yaw rate
                    let target_heading = Angle::between_points(player_pos, *target_pos);
                    input.with_yaw(target_heading);
                    input.with_angular_speed_limit(1.5);

                    // Check distance to target - kick when close enough
                    let distance_to_target = (player_pos - *target_pos).magnitude();
                    if distance_to_target < 50.0 {
                        println!(
                            "Move with ball done: distance_to_target<{:.2}",
                            distance_to_target
                        );
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Done;
                        return SkillProgress::Continue(input);
                    }

                    let ball_distance = (ball_pos - player_pos).magnitude();
                    if ball_distance > 200.0 {
                        println!("Failed to fetch ball: ball_distance={:.2}", ball_distance);
                        return SkillProgress::Done(SkillResult::Failure);
                    }

                    // Check if we've moved too far
                    let distance_moved = (player_pos - *start_pos).magnitude();
                    if distance_moved > 900.0 {
                        println!(
                            "Failed to move after fetch ball: distance_moved={:.2}",
                            distance_moved
                        );
                        input.with_kicker(Kick { force: 1.0 });
                        self.state = FetchBallWithPreshootState::Failed;
                        return SkillProgress::Continue(input);
                    }

                    input.with_position(*go_to_pos);
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
