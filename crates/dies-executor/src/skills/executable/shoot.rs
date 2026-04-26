//! ReflexShoot skill - orient toward target and kick.
//!
//! This is a discrete skill that rotates to face a target position
//! and kicks the ball.

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{KickerControlInput, PlayerControlInput, Velocity};

/// Angle tolerance for considering the robot "aligned" with target
const ALIGNMENT_TOLERANCE: f64 = 6.0 * std::f64::consts::PI / 180.0; // 6 degrees
/// Angular speed limit while turning with ball
const ANGULAR_SPEED_LIMIT: f64 = 1.3; // radians/s
/// Angular acceleration limit while turning with ball
const ANGULAR_ACCELERATION_LIMIT: f64 = 240.0 * std::f64::consts::PI / 180.0;
/// Dribbler speed during shot
const DRIBBLER_SPEED: f64 = 0.6;
/// Maximum distance from ball before declaring failure
const MAX_BALL_DISTANCE: f64 = 350.0;

/// State machine for reflex shoot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShootState {
    /// Rotating to face the target
    Facing,
    /// Aligned, ready to kick
    Kicking,
    /// Kick has been commanded
    KickCommanded,
}

/// A skill that orients toward a target and kicks.
///
/// This is a discrete skill - start once and monitor status. The skill
/// completes when the ball is kicked.
pub struct ShootSkill {
    target: Vector2,
    status: SkillStatus,
    state: ShootState,
}

impl ShootSkill {
    /// Create a new ReflexShoot skill.
    pub fn new(target: Vector2) -> Self {
        Self {
            target,
            status: SkillStatus::Running,
            state: ShootState::Facing,
        }
    }
}

impl ExecutableSkill for ShootSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::Shoot { .. })
    }

    fn update_params(&mut self, command: &SkillCommand) {
        if let SkillCommand::Shoot { target } = command {
            self.target = *target;
            // Note: We don't reset the state machine - just update the target
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        // Check if we still have the ball
        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_dist = (ball.position.xy() - ctx.player.position).norm();
            if ball_dist > MAX_BALL_DISTANCE {
                self.status = SkillStatus::Failed;
                return SkillProgress::failure();
            }
        }

        // Check breakbeam
        if !ctx.player.breakbeam_ball_detected && self.state != ShootState::KickCommanded {
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED);

        // Calculate target heading
        let target_heading = Angle::between_points(ctx.player.position, self.target);
        let heading_error = (ctx.player.yaw - target_heading).abs();

        match self.state {
            ShootState::Facing => {
                input.with_yaw(target_heading);
                input.with_care(0.7);
                input.with_angular_acceleration_limit(ANGULAR_ACCELERATION_LIMIT);
                input.with_angular_speed_limit(ANGULAR_SPEED_LIMIT);

                // Move slowly toward ball to maintain contact
                if let Some(ball) = ctx.world.ball.as_ref() {
                    let ball_pos = ball.position.xy();
                    let player_pos = ctx.player.position;
                    let dir = (ball_pos - player_pos).normalize();
                    input.velocity = Velocity::global(dir * 50.0);
                }

                if heading_error < ALIGNMENT_TOLERANCE {
                    self.state = ShootState::Kicking;
                }
            }
            ShootState::Kicking => {
                input.with_yaw(target_heading);
                input.with_kicker(KickerControlInput::Kick { force: 1.0 });
                self.state = ShootState::KickCommanded;
            }
            ShootState::KickCommanded => {
                // Kick has been commanded, wait for ball to leave
                // In practice, this usually succeeds immediately after kick
                self.status = SkillStatus::Succeeded;
                return SkillProgress::success();
            }
        }

        self.status = SkillStatus::Running;
        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }
}
