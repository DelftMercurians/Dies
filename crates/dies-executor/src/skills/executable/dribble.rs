//! Dribble skill - move to a position while carrying the ball.
//!
//! This is a continuous skill that moves the robot to a target position
//! with the ball, using the dribbler and limited acceleration.

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

const DEFAULT_POS_TOLERANCE: f64 = 50.0;
const DEFAULT_VEL_TOLERANCE: f64 = 20.0;
const DRIBBLE_ACCELERATION_LIMIT: f64 = 700.0;
const DRIBBLE_ANGULAR_ACCELERATION_LIMIT: f64 = 180.0_f64 * std::f64::consts::PI / 180.0;
const DRIBBLE_ANGULAR_SPEED_LIMIT: f64 = 180.0_f64 * std::f64::consts::PI / 180.0;
const DRIBBLER_SPEED: f64 = 1.0;

/// Enter pivot-around-ball phase when heading error exceeds this (radians).
const PIVOT_ENTRY_THRESHOLD: f64 = 25.0_f64 * std::f64::consts::PI / 180.0;
/// Exit pivot phase when heading error drops below this (radians). Hysteresis
/// prevents oscillation at the boundary.
const PIVOT_EXIT_THRESHOLD: f64 = 5.0_f64 * std::f64::consts::PI / 180.0;
/// Translational speed while pivoting around the ball (mm/s). Values matched
/// to the PR #48 implementation that was validated in simulation.
const PIVOT_SPEED: f64 = 300.0;
/// Distance from robot center to the ball while dribbling (mm). Tuned
/// empirically in PR #48; nominally ~111 mm (PLAYER_RADIUS + BALL_RADIUS)
/// but 140 mm was found to work better in sim.
const BALL_DISTANCE_FROM_ROBOT: f64 = 140.0;

/// A skill that moves the robot to a target position while carrying the ball.
///
/// This is a continuous skill - calling it repeatedly with different positions
/// will smoothly update the trajectory.
///
/// When the yaw error to `target_heading` is large, the skill first pivots
/// around the ball (translating perpendicular to its heading while rotating)
/// to bring the heading in line before translating toward `target_pos`. This
/// avoids the ball-loss that can happen when the low-level controller tries
/// to rotate and translate aggressively at the same time.
///
/// The skill fails immediately if the robot doesn't have the ball (breakbeam
/// not triggered).
pub struct DribbleSkill {
    target_pos: Vector2,
    target_heading: Angle,
    pos_tolerance: f64,
    vel_tolerance: f64,
    status: SkillStatus,
    pivoting: bool,
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
            pivoting: false,
        }
    }
}

impl ExecutableSkill for DribbleSkill {
    fn matches_command(&self, command: &SkillCommand) -> bool {
        matches!(command, SkillCommand::Dribble { .. })
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

        // Heading error sign matches PR #48: current minus target. Pivot
        // velocity and angular velocity signs are paired to this convention
        // — do not flip without running in sim; the command pipeline has
        // non-obvious sign conventions downstream.
        let yaw_err = (ctx.player.yaw - self.target_heading).radians();

        // Update pivoting state with hysteresis.
        if self.pivoting && yaw_err.abs() < PIVOT_EXIT_THRESHOLD {
            self.pivoting = false;
        } else if !self.pivoting && yaw_err.abs() > PIVOT_ENTRY_THRESHOLD {
            self.pivoting = true;
        }

        // Only declare arrival when heading is also aligned — otherwise the
        // caller's follow-up (e.g., ReflexShoot) would be set up wrong.
        if !self.pivoting
            && distance < self.pos_tolerance
            && speed < self.vel_tolerance
            && yaw_err.abs() < PIVOT_EXIT_THRESHOLD
        {
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        self.status = SkillStatus::Running;

        let mut input = PlayerControlInput::new();
        input.with_dribbling(DRIBBLER_SPEED);

        if self.pivoting {
            // Pivot around the ball: translate perpendicular to current
            // heading while rotating, so the ball stays at the dribbler.
            // Signs mirror PR #48 (tested in simulation).
            let dir_sign = yaw_err.signum();
            let v_local = Vector2::new(0.0, dir_sign * PIVOT_SPEED);
            let omega = dir_sign * PIVOT_SPEED / BALL_DISTANCE_FROM_ROBOT;
            input.velocity = Velocity::local(v_local);
            input.angular_velocity = Some(omega);
            input.with_angular_speed_limit(omega.abs() * 1.5);
        } else {
            input.with_position(self.target_pos);
            input.with_yaw(self.target_heading);
            // Use limited acceleration to avoid losing the ball
            input.with_acceleration_limit(DRIBBLE_ACCELERATION_LIMIT);
            input.with_angular_acceleration_limit(DRIBBLE_ANGULAR_ACCELERATION_LIMIT);
            input.with_angular_speed_limit(DRIBBLE_ANGULAR_SPEED_LIMIT);
        }

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }
}
