use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

#[derive(Clone)]
pub struct ReceiveSkill {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
    status: SkillStatus,
}

impl ReceiveSkill {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
            status: SkillStatus::Running,
        }
    }
}

impl ExecutableSkill for ReceiveSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::Receive { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::Receive {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        } = command
        {
            self.from_pos = *from_pos;
            self.target_pos = *target_pos;
            self.capture_limit = *capture_limit;
            self.cushion = *cushion;
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let breakbeam = ctx.player.breakbeam_ball_detected;
        if breakbeam {
            log::info!("ReceiveV2: Ball captured successfully");
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        let current_pos = ctx.player.position.xy();

        // Calculate angle to look at from_pos
        let look_direction = self.from_pos - current_pos;
        let target_heading = Angle::from_radians(look_direction.y.atan2(look_direction.x));

        // Move towards target
        //input.velocity = Velocity::Global(diff.normalize() * 100.0);
        input.with_dribbling(0.6);
        input.with_yaw(target_heading);

        // Project ball onto the normal line (perpendicular to from->target line, passing through target)
        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_pos = ball.position.xy();
            let ball_vel = ball.velocity.xy();

            // Calculate line direction and its perpendicular (normal)
            let line_vec = self.target_pos - self.from_pos;
            let normal = Vector2::new(-line_vec.y, line_vec.x).normalize();

            // Project ball onto the normal line (perpendicular to from->target)
            let to_ball = ball_pos - self.target_pos;
            let distance_along_normal = to_ball.dot(&normal);
            let distance_along_normal =
                distance_along_normal.clamp(-self.capture_limit, self.capture_limit);

            // Calculate ball's projected position on the normal line
            let ball_projection = self.target_pos + normal * distance_along_normal;

            // Claude calculation - super duper shit shit
            let mut target_position = ball_projection;

            // let dot_product = to_ball.dot(&normal) * ball_vel.dot(&normal);
            // log::info!("ball magnitude: {:.3}, velocity: ({:.3}, {:.3}), dot_product: {:.3}", ball.velocity.magnitude(), ball.velocity.x, ball.velocity.y, dot_product);
            // // If ball has significant velocity and is moving towards the line, predict intersection
            // if ball_vel.magnitude() > 1500.0 && dot_product < 0.0 {
            //     // Ball is moving away from line (towards it) - predict where it will intersect
            //     let line_direction = line_vec.normalize();
            //     let denominator = ball_vel.dot(&line_direction);

            //     if denominator.abs() > 1e-6 {
            //         let t = -to_ball.dot(&line_direction) / denominator;

            //         if t > 0.0 && t < 10.0 {
            //             let future_ball_pos = ball_pos + ball_vel * t;
            //             let to_future = future_ball_pos - self.target_pos;
            //             let intercept_distance = to_future.dot(&normal).clamp(-self.capture_limit, self.capture_limit);
            //             target_position = self.target_pos + normal * intercept_distance;
            //         }
            //     }
            // }

            input.with_position(target_position);
            // }
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }
}
