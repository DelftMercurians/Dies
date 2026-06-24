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

const DRIBBLING_DISTANCE: f64 = 300.0;
const BALL_MOVED_FAIL_DISTANCE: f64 = 100.0;
const DRIBBLER_SPEED: f64 = 0.6;

enum PickupState {
    Approaching,
    FinalApproach {
        last_good_ball_pos: Vector2,
        starting_position: Vector2,
    },
}

pub struct PickupBallSkill {
    target_heading: Angle,
    skill_status: SkillStatus,
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
            starting_position: None,
            breakbeam_on: None,
            state: PickupState::Approaching,
        }
    }

    /// Update the post-capture heading in place (used when composed inside the
    /// pass coordinator's `Secure` phase).
    pub fn set_target_heading(&mut self, target_heading: Angle) {
        self.target_heading = target_heading;
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

        // Check if breakbeam has been triggered
        if ctx.player.has_ball {
            self.skill_status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        match self.state {
            PickupState::Approaching => {
                let approach_dir = self.target_heading.to_vector();
                let approach_pos = ball_pos - approach_dir * (PLAYER_RADIUS + BALL_RADIUS + 80.0);
                dies_core::debug_cross(
                    "pickup_ball_target",
                    approach_pos,
                    dies_core::DebugColor::Blue,
                );

                input.with_position(approach_pos);
                input.with_yaw(self.target_heading);
                input.avoid_ball = true;
                input.avoid_ball_care = 0.0;

                let to_ball_heading = Angle::from_vector(ball_pos - player_pos);
                let cone_err = (ctx.player.yaw - to_ball_heading).abs();
                dies_core::debug_string(
                    "pickup_ball_state",
                    format!(
                        "approach: cone_err: {:.1}deg; distance: {:.1}",
                        cone_err, distance
                    ),
                );
                if cone_err < 10.0_f64.to_radians()
                    && distance < (PLAYER_RADIUS + BALL_RADIUS + 100.0)
                {
                    self.state = PickupState::FinalApproach {
                        last_good_ball_pos: ball_pos,
                        starting_position: player_pos,
                    };
                }
            }
            PickupState::FinalApproach {
                last_good_ball_pos,
                starting_position,
            } => {
                // If ball detected, update last good position and heading
                if ball.detected {
                    self.state = PickupState::FinalApproach {
                        last_good_ball_pos: ball_pos,
                        starting_position,
                    };
                    // Go back to approaching if the ball is too far away
                    let cone_err =
                        (ctx.player.yaw - Angle::from_vector(ball_pos - player_pos)).abs();
                    if cone_err > 10.0_f64.to_radians() {
                        self.state = PickupState::Approaching;
                    }
                }

                let approach_vector = if ball.detected {
                    ball_pos - ctx.player.position
                } else {
                    last_good_ball_pos - ctx.player.position
                };
                let approach_distance = approach_vector.norm();
                input.add_global_velocity(
                    approach_vector.normalize() * (approach_distance + 70.0) * 1.0,
                );
                input.with_yaw(Angle::from_vector(approach_vector));
                input.with_dribbling(DRIBBLER_SPEED);
                input.avoid_ball = false;
                dies_core::debug_string("pickup_ball_state", format!("final_approach"));

                let ball_dist = (ball_pos - last_good_ball_pos).norm();
                if ball_dist > BALL_MOVED_FAIL_DISTANCE
                    || (player_pos - starting_position).norm() > 300.0
                {
                    self.skill_status = SkillStatus::Failed;
                    return SkillProgress::failure();
                }
            }
        }

        self.skill_status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.skill_status
    }
}
