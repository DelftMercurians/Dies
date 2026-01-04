//! PickupBall skill - approach and capture the ball.
//!
//! This is a discrete skill that approaches the ball from behind and
//! captures it using the dribbler.

use std::time::{Duration, Instant};

use dies_core::{Angle, Vector2, BALL_RADIUS, PLAYER_RADIUS};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress, SkillType};
use crate::control::{PlayerControlInput, Velocity};

/// Distance from ball at which we start slowing down and using the dribbler
const DRIBBLING_DISTANCE: f64 = 1000.0;
/// Distance to maintain when close to a stationary ball
const STOP_DISTANCE: f64 = PLAYER_RADIUS + BALL_RADIUS + 30.0;
/// Maximum relative speed when approaching the ball
const MAX_RELATIVE_SPEED: f64 = 1000.0;
/// Dribbler speed during pickup
const DRIBBLER_SPEED: f64 = 0.6;
/// Time after breakbeam triggers before declaring success
const BREAKBEAM_CONFIRM_DURATION: Duration = Duration::from_millis(100);
/// Maximum distance we allow robot to move during final approach
const MAX_FINAL_APPROACH_DISTANCE: f64 = 30.0;

/// A skill that approaches and captures the ball.
///
/// This is a discrete skill - start once and monitor status. The skill
/// completes when the breakbeam detects the ball.
///
/// The `target_heading` parameter allows pre-orienting the robot for a
/// follow-up action (e.g., facing the goal before shooting).
pub struct PickupBallSkill {
    target_heading: Angle,
    status: SkillStatus,
    last_good_heading: Option<Angle>,
    starting_position: Option<Vector2>,
    breakbeam_on: Option<Instant>,
}

impl PickupBallSkill {
    /// Create a new PickupBall skill.
    pub fn new(target_heading: Angle) -> Self {
        Self {
            target_heading,
            status: SkillStatus::Running,
            last_good_heading: None,
            starting_position: None,
            breakbeam_on: None,
        }
    }
}

impl ExecutableSkill for PickupBallSkill {
    fn skill_type(&self) -> SkillType {
        SkillType::PickupBall
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::PickupBall { target_heading } = command {
            self.target_heading = *target_heading;
            // Note: We don't reset status here - the skill continues with updated params
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        let Some(ball) = ctx.world.ball.as_ref() else {
            // Wait for the ball to appear
            return SkillProgress::Continue(PlayerControlInput::default());
        };

        let ball_pos = ball.position.xy();
        let ball_speed = ball.velocity.xy().norm();
        let player_pos = ctx.player.position;
        let distance = (ball_pos - player_pos).norm();

        // Calculate heading toward ball
        let ball_angle = {
            let angle = Angle::between_points(player_pos, ball_pos);
            if distance > 50.0 {
                self.last_good_heading = Some(angle);
                angle
            } else {
                self.last_good_heading.unwrap_or(angle)
            }
        };

        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED);
        input.with_yaw(ball_angle);

        // Check if breakbeam has been triggered
        if ctx.player.breakbeam_ball_detected && distance < 400.0 {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        // Handle breakbeam confirmation with timeout
        if ctx.player.breakbeam_ball_detected {
            if self.breakbeam_on.is_none() {
                self.breakbeam_on = Some(Instant::now());
            }
        } else {
            self.breakbeam_on = None;
        }

        if let Some(breakbeam_on) = self.breakbeam_on {
            let elapsed = breakbeam_on.elapsed();
            if elapsed > BREAKBEAM_CONFIRM_DURATION {
                self.breakbeam_on = None;
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            } else {
                // Move slowly toward ball while waiting for confirmation
                let vel = (0.1 - elapsed.as_secs_f64()) / 0.1
                    * 100.0
                    * ball_angle.rotate_vector(&Vector2::x());
                input.velocity = Velocity::global(vel);
                return SkillProgress::Continue(input);
            }
        }

        if ball_speed < 100.0 {
            // Ball is stationary or slow - approach it
            if distance > STOP_DISTANCE + 80.0 {
                // Move close to the ball
                let target_pos =
                    ball_pos - ball_angle.rotate_vector(&Vector2::new(STOP_DISTANCE, 0.0));
                input.with_position(target_pos);
                input.with_care(0.8);
            } else {
                // Final approach - move slowly
                let start_pos = *self.starting_position.get_or_insert(player_pos);
                let moved_distance = (player_pos - start_pos).norm();

                if moved_distance > MAX_FINAL_APPROACH_DISTANCE {
                    // We've moved too far, probably stuck
                    self.status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }

                input.velocity = Velocity::global(
                    (1.0 / moved_distance.max(1.0))
                        * 100.0
                        * ball_angle.rotate_vector(&Vector2::x()),
                );
            }
        } else {
            // Ball is moving - intercept it
            // Sample points on the ball trajectory and find where we can intercept
            let points_schedule = [
                0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 1.0,
                1.2, 1.5, 2.0,
            ];
            let friction_factor = 1.5;

            let ball_points: Vec<Vector2> = points_schedule
                .iter()
                .map(|t| {
                    ball_pos
                        + ball.velocity.xy() * (*t) * (1.0 - f64::min(1.0, (*t) / friction_factor))
                })
                .collect();

            let mut intersection = ball_points[ball_points.len() - 1];
            for i in 0..ball_points.len() - 1 {
                let a = ball_points[i];
                let b = ball_points[i + 1];
                let must_be_reached_before = points_schedule[i];

                let mut time_to_reach = f64::min(
                    ctx.world.time_to_reach_point(ctx.player, a),
                    ctx.world.time_to_reach_point(ctx.player, b),
                );
                // Add margin for accuracy
                time_to_reach = time_to_reach * 1.2 + 0.1;

                if time_to_reach < must_be_reached_before {
                    intersection = b;
                    break;
                }
            }

            input.with_position(intersection);

            // Once close, use proportional control
            if distance < DRIBBLING_DISTANCE {
                input.position = None;
                input.velocity = Velocity::global(
                    (ball_speed * 0.8 + distance * (MAX_RELATIVE_SPEED / DRIBBLING_DISTANCE))
                        * (ball_pos - player_pos).normalize(),
                );
            }
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pickup_ball_creation() {
        let skill = PickupBallSkill::new(Angle::from_radians(0.0));

        assert_eq!(skill.skill_type(), SkillType::PickupBall);
        assert_eq!(skill.status(), SkillStatus::Running);
    }

    #[test]
    fn test_pickup_ball_update_params() {
        let mut skill = PickupBallSkill::new(Angle::from_radians(0.0));

        skill.update_params(&SkillCommand::PickupBall {
            target_heading: Angle::from_radians(1.5),
        });

        assert!((skill.target_heading.radians() - 1.5).abs() < 0.001);
    }
}
