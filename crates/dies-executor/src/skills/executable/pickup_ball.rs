//! PickupBall skill - approach and capture the ball.
//!
//! This is a discrete skill that approaches the ball from behind and
//! captures it using the dribbler.

use std::fmt::format;
use std::time::{Duration, Instant};

use dies_core::{Angle, Vector2, BALL_RADIUS, PLAYER_RADIUS};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

/// Distance from ball at which we start slowing down and using the dribbler
const DRIBBLING_DISTANCE: f64 = 1000.0;
/// Distance to maintain when close to a stationary ball
const STOP_DISTANCE: f64 = PLAYER_RADIUS + BALL_RADIUS + 10.0;
/// Maximum relative speed when approaching the ball
const MAX_RELATIVE_SPEED: f64 = 1000.0;
/// Dribbler speed during pickup
const DRIBBLER_SPEED: f64 = 0.2;
/// Time after breakbeam triggers before declaring success
const BREAKBEAM_CONFIRM_DURATION: Duration = Duration::from_millis(100);
/// Maximum distance we allow robot to move during final approach
const MAX_FINAL_APPROACH_DISTANCE: f64 = 80.0;

enum PickupState {
    Uninitialized,
    GoingToApproachPos,
    FinalApproach,
    Capturing,
}

pub struct PickupBallSkill {
    target_heading: Angle,
    skill_status: SkillStatus,
    last_good_heading: Option<Angle>,
    starting_position: Option<Vector2>,
    breakbeam_on: Option<Instant>,
    state: PickupState,
}

impl PickupBallSkill {
    /// Create a new PickupBall skill.
    pub fn new(target_heading: Angle) -> Self {
        Self {
            target_heading,
            skill_status: SkillStatus::Running,
            last_good_heading: None,
            starting_position: None,
            breakbeam_on: None,
            state: PickupState::Uninitialized,
        }
    }
}

impl ExecutableSkill for PickupBallSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::PickupBall { .. })
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
        dies_core::debug_value(
            format!("{}.ball_dist", ctx.debug_prefix),
            distance - PLAYER_RADIUS - BALL_RADIUS,
        );

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

        let has_ball =
            ctx.player.breakbeam_ball_detected || (distance - PLAYER_RADIUS - BALL_RADIUS) < -2.0;

        // Check if breakbeam has been triggered
        if has_ball {
            self.skill_status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        // Handle breakbeam confirmation with timeout
        // if ctx.player.breakbeam_ball_detected {
        //     if self.breakbeam_on.is_none() {
        //         self.breakbeam_on = Some(Instant::now());
        //     }
        // } else {
        //     self.breakbeam_on = None;
        // }

        if let Some(breakbeam_on) = self.breakbeam_on {
            let elapsed = breakbeam_on.elapsed();
            if elapsed > BREAKBEAM_CONFIRM_DURATION {
                self.breakbeam_on = None;
                self.skill_status = SkillStatus::Succeeded;
                return SkillProgress::success();
            } else {
                dies_core::debug_string(
                    format!("{}.pickup_ball.status", ctx.debug_prefix),
                    format!(
                        "breakbeam triggered, confirming... ({:.0} ms)",
                        elapsed.as_secs_f64() * 1000.0
                    ),
                );
                // Move slowly toward ball while waiting for confirmation
                let vel = (0.1 - elapsed.as_secs_f64()) / 0.1
                    * 100.0
                    * ball_angle.rotate_vector(&Vector2::x());
                input.velocity = Velocity::global(vel);
                return SkillProgress::Continue(input);
            }
        }

        if ball_speed < 100.0 {
            // Ball is stationary: approach from the side opposite to target_heading
            // so that post-capture the robot is already facing target_heading.
            let approach_dir = self.target_heading.to_vector();
            let approach_pos = ball_pos - approach_dir * STOP_DISTANCE;

            let dist_to_approach = (approach_pos - player_pos).norm();
            if dist_to_approach > STOP_DISTANCE + 15.0 {
                dies_core::debug_string(
                    format!("{}.pickup_ball.status", ctx.debug_prefix),
                    "go to pos",
                );
                input.with_position(approach_pos);
                input.with_yaw(self.target_heading);
                input.with_care(0.8);
                input.avoid_ball = true;
                input.avoid_ball_care = 1.5;
            } else {
                // Final approach - creep forward along target_heading
                let start_pos = *self.starting_position.get_or_insert(player_pos);
                let moved_distance = (player_pos - start_pos).norm();

                // if moved_distance > MAX_FINAL_APPROACH_DISTANCE {
                //     self.status = SkillStatus::Failed;
                //     return SkillProgress::failure();
                // }

                let global_approach_vel = (1.0 / moved_distance.max(1.0)) * 3000.0 * approach_dir;
                input.velocity = Velocity::global(global_approach_vel);
                dies_core::debug_string(
                    format!("{}.pickup_ball.status", ctx.debug_prefix),
                    format!(
                        "approach ball: {:.1}, {:.1}",
                        global_approach_vel[0], global_approach_vel[1]
                    ),
                );
                input.with_yaw(self.target_heading);
                dies_core::debug_line(
                    format!("{}.pickup_ball.approach_line", ctx.debug_prefix),
                    player_pos,
                    player_pos + global_approach_vel,
                    dies_core::DebugColor::Orange,
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

        self.skill_status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.skill_status
    }
}
