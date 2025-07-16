use std::time::{Duration, Instant};

use dies_core::{distance_to_line, find_intersection, perp, Angle, Vector2};

use crate::{
    skills::{SkillCtx, SkillProgress, SkillResult},
    PlayerControlInput,
};

pub struct TryReceive {
    started: bool,
    wait_start: Option<Instant>,
    waiting_for_velocity: bool,
    intercept_line: Option<(Vector2, Vector2)>,
}

impl TryReceive {
    pub fn new() -> Self {
        Self {
            started: false,
            wait_start: None,
            waiting_for_velocity: true,
            intercept_line: None,
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if !self.started && ctx.bt_context.take_passing_target(ctx.player.id).is_some() {
            println!("receive started");
            self.started = true;
        } else if !self.started {
            return SkillProgress::Done(SkillResult::Failure);
        }

        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Done(SkillResult::Failure);
        };
        let to_robot = ctx.player.position - ball.position.xy();
        let ball_vel = ball.velocity.xy();
        let ball_pos = ball.position.xy();

        let mut input = PlayerControlInput::new();
        // always face the ball
        input.with_yaw(Angle::between_points(
            ctx.player.position,
            ball.position.xy(),
        ));
        input.with_dribbling(1.0); // immediate dribbling

        if self.waiting_for_velocity {
            let wait_start = self.wait_start.get_or_insert(Instant::now());
            let wait_time = wait_start.elapsed();
            if wait_time > Duration::from_secs(3) {
                println!("receive failed: waited too long");
                return SkillProgress::Done(SkillResult::Failure);
            }

            // If ball is nearly stationary, treat as not coming
            if ball_vel.norm() < 50.0 {
                return SkillProgress::Continue(input);
            }

            // Otherwise the ball is on the way
            self.waiting_for_velocity = false;
        }

        let to_robot_dir = to_robot.normalize();
        let ball_vel_dir = ball_vel.normalize();
        let angle = Angle::between_points(to_robot_dir, ball_vel_dir);
        // if angle.degrees().abs() > 30.0 && ball_vel.norm() > 50.0 {
        //     println!("receive failed: ball is not coming towards us");
        //     return SkillProgress::Done(SkillResult::Failure);
        // }

        if ball_vel.norm() < 50.0 {
            println!("receive failed: ball is not moving");
            return SkillProgress::Done(SkillResult::Failure);
        }

        let line_dist = distance_to_line(ball_pos, ball_pos + to_robot, ball_pos);
        // if the ball is too far away, give up
        if line_dist > 400.0 {
            println!("receive failed: ball is too far away");
            return SkillProgress::Done(SkillResult::Failure);
        }

        let closest_opponent_dist = ctx
            .world
            .opp_players
            .iter()
            .map(|p| (ball_pos - p.position).norm() as i64)
            .min();
        // if the closest opponent is too close, give up
        if let Some(closest_opponent_dist) = closest_opponent_dist {
            if closest_opponent_dist < 120 {
                println!("receive failed: closest opponent is too close");
                return SkillProgress::Done(SkillResult::Failure);
            }
        }

        let intercept_line = self
            .intercept_line
            .get_or_insert((ctx.player.position, perp(ball.velocity.xy())));

        if let Some(intersection) = find_intersection(
            intercept_line.0,
            intercept_line.1,
            ball.position.xy(),
            ball.velocity.xy(),
        ) {
            // Clamp the intersection within 500mm of the starting position
            let start_pos = intercept_line.0;
            let to_intersection = intersection - start_pos;
            let clamped_intersection = if to_intersection.magnitude() > 500.0 {
                start_pos + to_intersection.normalize() * 500.0
            } else {
                intersection
            };
            input.with_position(clamped_intersection);

            // if ctx.player.breakbeam_ball_detected {
            //     println!("receive success");
            //     SkillProgress::Done(SkillResult::Success)
            // } else {
            SkillProgress::Continue(input)
            // }
        } else {
            println!("receive failed: no intersection");
            SkillProgress::failure()
        }
    }
}
