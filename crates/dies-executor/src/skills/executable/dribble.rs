//! Dribble skill - move to a position while carrying the ball.
//!
//! This is a continuous skill that moves the robot to a target position
//! with the ball, using the dribbler and limited acceleration.

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress, SkillType};
use crate::control::PlayerControlInput;

const DEFAULT_POS_TOLERANCE: f64 = 50.0;
const DEFAULT_VEL_TOLERANCE: f64 = 20.0;
const DRIBBLE_ACCELERATION_LIMIT: f64 = 700.0;
const DRIBBLE_ANGULAR_ACCELERATION_LIMIT: f64 = 180.0_f64 * std::f64::consts::PI / 180.0;
const DRIBBLE_ANGULAR_SPEED_LIMIT: f64 = 180.0_f64 * std::f64::consts::PI / 180.0;
const DRIBBLER_SPEED: f64 = 1.0;

/// A skill that moves the robot to a target position while carrying the ball.
///
/// This is a continuous skill - calling it repeatedly with different positions
/// will smoothly update the trajectory.
///
/// The skill fails immediately if the robot doesn't have the ball (breakbeam
/// not triggered).
pub struct DribbleSkill {
    target_pos: Vector2,
    target_heading: Angle,
    pos_tolerance: f64,
    vel_tolerance: f64,
    status: SkillStatus,
}

impl DribbleSkill {
    /// Create a new Dribble skill.
    pub fn new(target_pos: Vector2, target_heading: Angle) -> Self {
        Self {
            target_pos,
            target_heading,
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            vel_tolerance: DEFAULT_VEL_TOLERANCE,
            status: SkillStatus::Running,
        }
    }
}

impl ExecutableSkill for DribbleSkill {
    fn skill_type(&self) -> SkillType {
        SkillType::Dribble
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::Dribble {
            target_pos,
            target_heading,
        } = command
        {
            self.target_pos = *target_pos;
            self.target_heading = *target_heading;
            // Reset status to Running if we were completed
            if matches!(self.status, SkillStatus::Succeeded | SkillStatus::Failed) {
                self.status = SkillStatus::Running;
            }
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        // Check if we have the ball
        if !ctx.player.breakbeam_ball_detected {
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let position = ctx.player.position;
        let velocity = ctx.player.velocity;

        let distance = (self.target_pos - position).norm();
        let speed = velocity.norm();

        // Check if we've arrived
        if distance < self.pos_tolerance && speed < self.vel_tolerance {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        self.status = SkillStatus::Running;

        let mut input = PlayerControlInput::new();
        input.with_position(self.target_pos);
        input.with_yaw(self.target_heading);
        input.with_dribbling(DRIBBLER_SPEED);

        // Use limited acceleration to avoid losing the ball
        input.with_acceleration_limit(DRIBBLE_ACCELERATION_LIMIT);
        input.with_angular_acceleration_limit(DRIBBLE_ANGULAR_ACCELERATION_LIMIT);
        input.with_angular_speed_limit(DRIBBLE_ANGULAR_SPEED_LIMIT);

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
    fn test_dribble_creation() {
        let skill = DribbleSkill::new(
            Vector2::new(1000.0, 500.0),
            Angle::from_radians(0.0),
        );

        assert_eq!(skill.skill_type(), SkillType::Dribble);
        assert_eq!(skill.status(), SkillStatus::Running);
    }

    #[test]
    fn test_dribble_update_params() {
        let mut skill = DribbleSkill::new(
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );

        skill.update_params(&SkillCommand::Dribble {
            target_pos: Vector2::new(1000.0, 1000.0),
            target_heading: Angle::from_radians(0.5),
        });

        assert_eq!(skill.target_pos, Vector2::new(1000.0, 1000.0));
    }
}

