use dies_core::{Angle, Vector2};

use crate::{control::Velocity, PlayerControlInput};
use super::{SkillCtx, SkillProgress};

#[derive(Clone)]
// A skill that receives the ball by moving left or right to the target position and capturing the ball
// with capture_limit being the distance from the target position
// cushion is whether to move in the same direction as the ball to cushion the ball
pub struct RecieveV2 {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
}

impl RecieveV2 {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        }
    }
    
    //TODO: Add cushioning if you are already at the correct intercept point and ball is coming in fast
    //TODO: add deadzone around intercept point where you just wait for the ball to come in instead of trying to move to it perfectly
    //TODO: FiX case when to fail and also why its nopt moving perpidicular to the from, target line
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
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
            let distance_along_normal = distance_along_normal.clamp(-self.capture_limit, self.capture_limit);

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
}