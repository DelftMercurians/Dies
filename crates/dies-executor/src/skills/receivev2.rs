use dies_core::{Angle, Vector2};

use crate::{control::Velocity, PlayerControlInput};
use super::{SkillCtx, SkillProgress};


const TARGET_DEADZONE: f64 = 100.0;
const DAMPING_TRIGGER_DISTANCE: f64 = 500.0; // Distance at which damping starts
const DAMPING_MIN_SPEED: f64 = 500.0; // Minimum ball speed to trigger damping (mm/s)
const DAMPING_COEFFICIENT: f64 = 0.2; // How much to dampen (0.0-1.0): lower = more damping

#[derive(Clone)]
// A skill that receives the ball by moving left or right to the target position and capturing the ball
// with capture_limit being the distance from the target position
// cushion is whether to move in the same direction as the ball to cushion the ball
pub struct RecieveV2 {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
    damping_active: bool,
}

impl RecieveV2 {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
            damping_active: false,
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
            
            // Claude calculation - smart verified
            let mut target_position = ball_projection;
            
            // Ball is moving away from line (towards it) - predict where it will intersect
            let line_direction = line_vec.normalize();
            let denominator = ball_vel.dot(&line_direction);
            
            if denominator.abs() > 1e-6 {
                let t = -to_ball.dot(&line_direction) / denominator;
                
                if t > 0.0 && t < 10.0 {
                    let future_ball_pos = ball_pos + ball_vel * t;
                    let to_future = future_ball_pos - self.target_pos;
                    let intercept_distance = to_future.dot(&normal).clamp(-self.capture_limit, self.capture_limit);
                    target_position = self.target_pos + normal * intercept_distance;
                }
            }

            // Check damping conditions
            let dist_to_ball = (ball_pos - current_pos).norm();
            let ball_speed = ball_vel.norm();
            let should_damping_continue = dist_to_ball < DAMPING_TRIGGER_DISTANCE && ball_speed > DAMPING_MIN_SPEED;
            
            if self.cushion && should_damping_continue {
                // Direction from robot to ball
                let to_ball_dir = (ball_pos - current_pos).normalize();
                
                // How much of the ball velocity is directed toward us
                let approach_velocity = -ball_vel.dot(&to_ball_dir); // Negative means approaching
                
                if approach_velocity > 0.0 {
                    // Ball is approaching! Activate damping
                    self.damping_active = true;
                    
                    // Calculate damping velocity
                    let damping_velocity = approach_velocity * DAMPING_COEFFICIENT;
                    let damping_direction = -to_ball_dir; // Move away from ball
                    let damping_vector = damping_direction * damping_velocity;
                    
                    log::info!("ReceiveV2: Ball damping active - dist: {:.1}mm, approach_vel: {:.1}mm/s, damping_vel: {:.1}mm/s", 
                        dist_to_ball, approach_velocity, damping_velocity);
                    
                    input.velocity = Velocity::Global(damping_vector);
                } else {
                    // Ball no longer approaching
                    self.damping_active = false;
                }
            } else {
                // Damping conditions no longer met
                self.damping_active = false;
            }

            let dist_to_target = (target_position - current_pos).norm();
            if dist_to_target < TARGET_DEADZONE || self.damping_active {
                log::info!("ReceiveV2: Within deadzone, waiting for ball to come in");
                return SkillProgress::Continue(input);
            }
            
            input.with_position(target_position);
        }

        SkillProgress::Continue(input)
    }
}