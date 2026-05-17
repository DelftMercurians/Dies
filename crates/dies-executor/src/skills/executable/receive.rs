use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

const TARGET_DEADZONE: f64 = 100.0; // Radius around target position to consider "arrived" (mm)
const DAMPING_TRIGGER_DISTANCE: f64 = 500.0; // Distance at which damping starts(mm)
const DAMPING_MIN_SPEED: f64 = 500.0; // Minimum ball speed to trigger damping (mm/s)
const DAMPING_COEFFICIENT: f64 = 0.2; // How much to dampen (0.0-1.0): lower = more damping

#[derive(Clone)]
pub struct ReceiveSkill {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
    damping_active: bool,
    status: SkillStatus,
}

impl ReceiveSkill {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
            damping_active: false,
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

            // Claude calculation - verified works
            let mut target_position = ball_projection;

            // Ball is moving away from line (towards it) - predict where it will intersect
            let line_direction = line_vec.normalize();
            let denominator = ball_vel.dot(&line_direction);

            if denominator.abs() > 1e-6 {
                let t = -to_ball.dot(&line_direction) / denominator;

                if t > 0.0 && t < 10.0 {
                    let future_ball_pos = ball_pos + ball_vel * t;
                    let to_future = future_ball_pos - self.target_pos;
                    let intercept_distance = to_future
                        .dot(&normal)
                        .clamp(-self.capture_limit, self.capture_limit);
                    target_position = self.target_pos + normal * intercept_distance;
                }
            }

            // Check damping conditions
            let dist_to_ball = (ball_pos - current_pos).norm();
            let ball_speed = ball_vel.norm();
            let should_damping_continue = dist_to_ball < DAMPING_TRIGGER_DISTANCE && ball_speed > DAMPING_MIN_SPEED;
            
            if self.cushion && should_damping_continue {
                let to_ball_dir = (ball_pos - current_pos).normalize();
                
                // How much of the ball velocity is directed toward us
                let approach_velocity = -ball_vel.dot(&to_ball_dir); // Negative means approaching
                
                if approach_velocity > 0.0 {
                    // Set activate damping to not escape it due to getting out of deadzone
                    self.damping_active = true;
                    
                    // Calculate damping velocity
                    let damping_velocity = approach_velocity * DAMPING_COEFFICIENT;
                    let damping_direction = -to_ball_dir; // Move away from ball
                    let damping_vector = damping_direction * damping_velocity;
                    
                    // log::info!("ReceiveV2: Ball damping active - dist: {:.1}mm, approach_vel: {:.1}mm/s, damping_vel: {:.1}mm/s", 
                    //     dist_to_ball, approach_velocity, damping_velocity);
                    
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
                // log::info!("Receive: Within deadzone, waiting for ball to come in");
                return SkillProgress::Continue(input);
            }

            input.with_position(target_position);
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }
}
