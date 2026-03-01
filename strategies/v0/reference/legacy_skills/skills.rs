use std::time::Instant;

use dies_core::{
    find_intersection, perp, which_side_of_robot, Angle, PlayerId, SysStatus, Vector2,
};

use super::{SkillCtx, SkillProgress};
use crate::{control::Velocity, skills::SkillResult, KickerControlInput, PlayerControlInput};

const DEFAULT_POS_TOLERANCE: f64 = 20.0;
const DEFAULT_VEL_TOLERANCE: f64 = 10.0;

#[derive(Clone)]
pub struct InterceptBall {
    intercept_line: Option<(Vector2, Vector2)>,
}

impl InterceptBall {
    pub fn new() -> Self {
        Self {
            intercept_line: None,
        }
    }
}

impl InterceptBall {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let intercept_line = self
                .intercept_line
                .get_or_insert((ctx.player.position, perp(ball.velocity.xy())));

            if let Some(intersection) = find_intersection(
                intercept_line.0,
                intercept_line.1,
                ball.position.xy(),
                ball.velocity.xy(),
            ) {
                let mut input = PlayerControlInput::new();

                input.with_position(intersection);
                input.with_yaw(Angle::between_points(
                    ctx.player.position,
                    ball.position.xy(),
                ));

                SkillProgress::Continue(input)
            } else {
                SkillProgress::failure()
            }
        } else {
            SkillProgress::failure()
        }
    }
}

#[derive(Clone)]
pub struct ApproachBall {}

impl ApproachBall {
    pub fn new() -> Self {
        Self {}
    }
}

impl ApproachBall {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position;

            if ctx.player.breakbeam_ball_detected
                && ctx
                    .world
                    .ball
                    .as_ref()
                    .map(|b| b.detected && ((b.position.xy() - ctx.player.position).norm() < 200.0))
                    .unwrap_or(true)
            {
                return SkillProgress::success();
            }

            let ball_speed = ball.velocity.xy().norm();
            let distance = (ball_pos - player_pos).norm();

            let mut input = PlayerControlInput::new();
            if (ball_pos - player_pos).norm() > 50.0 && ball.detected {
                input.with_yaw(Angle::between_points(player_pos, ball_pos));
            }

            let max_relative_speed = 800.0;
            let dribbling_distance = 1000.0;

            input.velocity = Velocity::global(
                (ball_speed * 0.8 + distance * (max_relative_speed / dribbling_distance))
                    * (ball_pos - player_pos).normalize(),
            );
            input.with_dribbling(1.0);

            SkillProgress::Continue(input)
        } else {
            SkillProgress::failure()
        }
    }
}

/// A skill that shoots the ball towards a target position with calculated force
#[derive(Clone)]
pub struct Shoot {
    target_position: Vector2,
    max_speed: f64,
    has_kicked: bool,
    timer: Option<Instant>,
    required_heading: Option<Angle>,
    heading_tolerance: f64,
}

impl Shoot {
    pub fn new(target_position: Vector2, max_speed: f64) -> Self {
        Self {
            target_position,
            max_speed,
            has_kicked: false,
            timer: None,
            required_heading: None,
            heading_tolerance: 0.1, // 0.1 radians tolerance for heading
        }
    }

    pub fn with_heading_tolerance(mut self, tolerance: f64) -> Self {
        self.heading_tolerance = tolerance;
        self
    }

    /// Calculate the required kick force based on distance and desired speed
    fn calculate_kick_force(&self, distance: f64) -> f64 {
        let base_force = 0.5;
        let distance_factor = distance / 1000.0;
        let speed_factor = self.max_speed / 1000.0;

        let force = base_force + (distance_factor * speed_factor * 2.0);

        force.min(5.0).max(0.1)
    }
}

impl Shoot {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();

        // Get ball information
        let ball = match ctx.world.ball.as_ref() {
            Some(ball) if ball.detected => ball,
            _ => return SkillProgress::failure(), // No ball detected
        };

        let player_pos = ctx.player.position;
        let ball_pos = ball.position.xy();

        // Check if we have the ball (should be close to the robot)
        let ball_distance = (ball_pos - player_pos).norm();
        if ball_distance > 250.0 {
            return SkillProgress::failure(); // Ball is too far
        }

        // Calculate required heading to target
        let target_direction = (self.target_position - player_pos).normalize();
        let required_heading = Angle::from_vector(target_direction);
        self.required_heading = Some(required_heading);

        // Check if we've already kicked
        if self.has_kicked {
            let timer = self.timer.get_or_insert(Instant::now());
            if timer.elapsed().as_secs_f64() > 0.2 {
                // Check if ball has moved away (indicating successful kick)
                if ball_distance > 300.0 {
                    return SkillProgress::success();
                } else {
                    return SkillProgress::failure();
                }
            } else {
                return SkillProgress::Continue(input);
            }
        }

        // Check if we're facing the right direction
        let heading_error = (ctx.player.yaw - required_heading).abs();
        if heading_error > self.heading_tolerance {
            // Still need to turn towards target
            input.with_yaw(required_heading);
            input.with_dribbling(1.0); // Keep ball while turning
            input.with_angular_acceleration_limit(240.0f64.to_radians());
            input.with_angular_speed_limit(180.0f64.to_radians());
            return SkillProgress::Continue(input);
        }

        // We're aligned with the target, now check kicker status
        let ready = match ctx.player.kicker_status.as_ref() {
            Some(SysStatus::Ready) => true,
            _ => false,
        };

        if ready && ctx.player.breakbeam_ball_detected {
            // Calculate distance to target and required force
            let distance_to_target = (self.target_position - player_pos).norm();
            let kick_force = self.calculate_kick_force(distance_to_target);

            // Execute the kick
            input.with_kicker(KickerControlInput::Kick { force: kick_force });
            input.kick_speed = Some(kick_force);
            self.has_kicked = true;
            SkillProgress::Continue(input)
        } else {
            // Arm the kicker and maintain dribbling
            input.kicker = KickerControlInput::Arm;
            input.with_dribbling(1.0);
            SkillProgress::Continue(input)
        }
    }
}
