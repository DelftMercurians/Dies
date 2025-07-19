use std::time::{Duration, Instant};

use dies_core::{distance_to_line, find_intersection, perp, Angle, Vector2};

use crate::{
    behavior_tree::PassingTarget,
    skills::{SkillCtx, SkillProgress, SkillResult},
    PlayerControlInput,
};

pub struct TryReceive {
    started: bool,
    wait_start: Option<Instant>,
    intercept_line: Option<(Vector2, Vector2)>,
    passer_position: Option<Vector2>,
    // passer_heading: Option<Angle>,
    passer_target: Option<Vector2>,
    success_timer_start: Option<Instant>,
}

impl TryReceive {
    pub fn new() -> Self {
        Self {
            started: false,
            wait_start: None,
            intercept_line: None,
            passer_position: None,
            // passer_heading: None,
            passer_target: None,
            success_timer_start: None,
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let Some(ball) = ctx.world.ball.as_ref() else {
            return SkillProgress::Done(SkillResult::Failure);
        };
        let ball_pos = ball.position.xy();
        let ball_distance = (ball_pos - ctx.player.position).norm();
        if ball_distance < 200.0 {
            let success_start_time = self
                .success_timer_start
                .get_or_insert(Instant::now())
                .clone();
            if success_start_time.elapsed().as_secs_f64() > 0.2 {
                println!("receive: ball is close, timer done");
                return SkillProgress::Done(SkillResult::Success);
            }
        } else if self.success_timer_start.is_some() {
            println!("receive: ball was close, it moved");
            return SkillProgress::Done(SkillResult::Success);
        }

        let passing_target = ctx.bt_context.take_passing_target(ctx.player.id);
        if passing_target.is_none() && self.passer_position.is_some() {
            let wait_start = self.wait_start.get_or_insert(Instant::now()).clone();
            if wait_start.elapsed() > Duration::from_secs(1) {
                if ball_distance < 500.0 {
                    println!("receive success: waited too long");
                    return SkillProgress::Done(SkillResult::Success);
                } else {
                    println!("receive failed: ball moved too far away");
                    return SkillProgress::Done(SkillResult::Failure);
                }
            }
        } else if let Some(passing_target) = passing_target.as_ref() {
            let passer_id = passing_target.shooter_id;
            if self.passer_position.is_none() {
                println!("receive: passer_id={:?}", passer_id);
            }
            self.passer_position = Some(ctx.world.get_player(passer_id).position);
            // self.passer_heading = Some(Angle::between_points(
            //     ctx.player.position,
            //     ctx.world.get_player(passer_id).position,
            // ));
            self.passer_target = Some(passing_target.position);
        } else {
            return SkillProgress::failure();
        }

        let mut input = PlayerControlInput::new();
        // always face the ball
        input.with_yaw(Angle::between_points(
            ctx.player.position,
            ball.position.xy(),
        ));
        // input.with_dribbling(1.0); // immediate dribbling

        let (Some(passer_position), Some(passer_target)) =
            (self.passer_position, self.passer_target)
        else {
            println!("recv ?");
            return SkillProgress::Continue(input);
        };

        // if self.waiting {
        //     let wait_start = self.wait_start.get_or_insert(Instant::now());
        //     let wait_time = wait_start.elapsed();
        //     if wait_time > Duration::from_secs(1) {
        //         println!("receive failed: waited too long");
        //         return SkillProgress::Done(SkillResult::Failure);
        //     }
        //     // If ball is nearly stationary, treat as not coming
        //     if ball_vel.norm() < 50.0 {
        //         return SkillProgress::Continue(input);
        //     }
        //     // Otherwise the ball is on the way
        //     self.waiting = false;
        // }
        // let to_robot_dir = to_robot.normalize();
        // let ball_vel_dir = ball_vel.normalize();
        // let angle = Angle::between_points(to_robot_dir, ball_vel_dir);
        // if angle.degrees().abs() > 30.0 && ball_vel.norm() > 50.0 {
        //     println!("receive failed: ball is not coming towards us");
        //     return SkillProgress::Done(SkillResult::Failure);
        // }
        // if ball_vel.norm() < 50.0 {
        //     println!("receive failed: ball is not moving");
        //     return SkillProgress::Done(SkillResult::Failure);
        // }

        // let line_dist = distance_to_line(ball_pos, ball_pos + to_robot, ball_pos);
        // // if the ball is too far away, give up
        // if line_dist > 400.0 {
        //     println!("receive failed: ball moved too far away");
        //     return SkillProgress::Done(SkillResult::Failure);
        // }

        let closest_opponent_dist = ctx
            .world
            .opp_players
            .iter()
            .map(|p| (ball_pos - p.position).norm() as i64)
            .min();
        // if the closest opponent is too close, give up
        if let Some(closest_opponent_dist) = closest_opponent_dist {
            if closest_opponent_dist < 60 {
                println!("receive failed: closest opponent is too close");
                return SkillProgress::Done(SkillResult::Failure);
            }
        }

        let ball_speed = ball.velocity.xy().norm();
        let intercept_line = (
            passer_target,
            perp((passer_position - passer_target).normalize()),
        );
        // .intercept_line
        // .get_or_insert((ctx.player.position, perp(passer_heading.to_vector())));

        ctx.team_context.debug_line_colored(
            "intercept_line",
            intercept_line.0 - intercept_line.1 * 200.0,
            intercept_line.0 + intercept_line.1 * 200.0,
            dies_core::DebugColor::Red,
        );
        ctx.team_context.debug_line_colored(
            "passing_line",
            passer_position,
            ctx.player.position,
            dies_core::DebugColor::Blue,
        );
        ctx.team_context.debug_cross_colored(
            "passer_position",
            passer_position,
            dies_core::DebugColor::Blue,
        );
        ctx.team_context.debug_cross_colored(
            "passer_target",
            passer_target,
            dies_core::DebugColor::Blue,
        );

        println!("receive: ball_speed {:?}", ball_speed);
        // let dir = if ball_speed > 100.0 {
        //     ball.velocity.xy()
        // } else {
        // passer_position - passer_target
        input.with_position(passer_target);
        input.aggressiveness = 3.0;
        return SkillProgress::Continue(input);
        // };
        // if let Some(intersection) =
        //     find_intersection(intercept_line.0, intercept_line.1, ball.position.xy(), dir)
        // {
        //     // println!("receive: intersection {:?}", intersection);
        //     // Clamp the intersection within 500mm of the starting position
        //     // let start_pos = intercept_line.0;
        //     // let to_intersection = intersection - start_pos;
        //     // let clamped_intersection = if to_intersection.magnitude() > 200.0 {
        //     //     start_pos + to_intersection.normalize() * 200.0
        //     // } else {
        //     //     intersection
        //     // };

        //     ctx.team_context.debug_cross_colored(
        //         "receive_target",
        //         intersection,
        //         dies_core::DebugColor::Purple,
        //     );

        //     input.with_position(intersection);
        //     input.aggressiveness = 3.0;

        //     // println!("receive positioing {:?}", clamped_intersection);
        //     SkillProgress::Continue(input)
        // } else {
        //     println!("receive failed: no intersection");
        //     SkillProgress::failure()
        // }
    }
}
