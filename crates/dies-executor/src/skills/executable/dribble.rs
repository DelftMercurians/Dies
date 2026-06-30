//! Dribble skill - move to a position while carrying the ball.
//!
//! This is a continuous skill that moves the robot to a target position
//! with the ball, using the dribbler and limited acceleration.

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};
use dies_tunables_macro::tunables;

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::{PlayerControlInput, Velocity};

tunables! {
    section "Dribble";

    /// Arrival position tolerance for the carry target.
    #[tunable(unit = "mm", min = 15.0, max = 300.0, step = 5.0)]
    DEFAULT_POS_TOLERANCE: f64 = 50.0;
    /// Heading tolerance to count as aligned (degrees).
    #[tunable(unit = "deg", min = 1.0, max = 30.0, step = 1.0)]
    DEFAULT_YAW_TOLERANCE: f64 = 5.0;
    /// Translational acceleration cap while carrying — gentle so the ball isn't
    /// shaken loose.
    #[tunable(unit = "mm/s²", min = 100.0, max = 2000.0, step = 50.0)]
    ACCELERATION_LIMIT: f64 = 500.0;
    /// Lateral speed while pivoting around the ball to fix heading.
    #[tunable(unit = "mm/s", min = 50.0, max = 800.0, step = 25.0)]
    ROTATE_AROUND_BALL_SPEED: f64 = 300.0;
    /// Centre-of-robot to centre-of-ball distance (≈111 mm by measurement).
    #[tunable(unit = "mm", min = 90.0, max = 200.0, step = 5.0)]
    BALL_TO_ROBOT_DISTANCE: f64 = 140.0;
    /// Angular speed cap while carrying.
    #[tunable(unit = "rad/s", min = 0.5, max = 4.0, step = 0.1)]
    MAX_ANGULAR_SPEED: f64 = 1.5;
}

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
    status: SkillStatus,
    target_pos: Vector2,
    target_heading: Angle,
    with_ball: bool,
    last_acceleration: f64,
    last_angular_acceleration: f64,
}

impl DribbleSkill {
    pub fn new(target_pos: Vector2, target_heading: Angle) -> Self {
        Self {
            status: SkillStatus::Running,
            target_pos,
            target_heading,
            with_ball: true,
            last_acceleration: 0.0,
            last_angular_acceleration: 0.0,
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
        }
    }

    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress {
        // log::info!("Dribbling towards position: {:?}, heading: {:?}, with_ball: {}", self.target_pos, self.target_heading, self.with_ball);
        // Check whether the robot holds the ball (unified possession signal).
        if !ctx.player.has_ball {
            log::warn!("Dribble skill failed: ball not captured");
            self.status = SkillStatus::Failed;
            return SkillProgress::failure();
        }

        let mut input = PlayerControlInput::new();
        let player_pos = ctx.player.position;
        let to_target = player_pos - self.target_pos;

        //Dribble around ball set by velocity control
        //First rotates to face target position then moves
        let heading_err =
            Angle::from_degrees(ctx.player.yaw.degrees() - self.target_heading.degrees());
        if self.with_ball == false && heading_err.degrees().abs() > DEFAULT_YAW_TOLERANCE() {
            // Velocity 90 degrees to the left or right of current heading
            let direction = if heading_err.radians() > 0.0 {
                Vector2::new(0.0, 1.0) // Rotate to the left
            } else {
                Vector2::new(0.0, -1.0) // Rotate to the right
            };
            //let direction = Vector2::new(0.0, 1.0);
            input.velocity = Velocity::local(direction * ROTATE_AROUND_BALL_SPEED());
            input.angular_velocity =
                Some(direction.y * ROTATE_AROUND_BALL_SPEED() / BALL_TO_ROBOT_DISTANCE()); // v = w*d and d is distance from the centre of robot to ball
            return SkillProgress::Continue(input);
        }

        self.with_ball = true; // switch to normal dribbling once heading is correct
        input.with_dribbling(0.5);

        input.with_position(self.target_pos);
        input.with_yaw(self.target_heading);
        //log::info!("cur_vel{:.2}, exp_vel{:.2}", current_velocity.magnitude(), ACCELERATION_LIMIT*distance_to_target);

        input.with_acceleration_limit(ACCELERATION_LIMIT());
        input.with_angular_speed_limit(MAX_ANGULAR_SPEED());
        // input.with_angular_acceleration_limit(ANGULAR_ACCELERATION_LIMIT);

        //TODO: implement rotating around ball when with_ball = false

        //        log::info!("to_target magnitude: {:.2}", to_target.magnitude());
        if to_target.magnitude() < DEFAULT_POS_TOLERANCE()
            && heading_err.degrees().abs() < DEFAULT_YAW_TOLERANCE()
        {
            log::info!("Dribble: At the target position with correct heading.");
            self.status = SkillStatus::Succeeded;
            return SkillProgress::success();
        }

        return SkillProgress::Continue(input);
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "Dribble"
    }

    fn description(&self) -> String {
        format!(
            "carrying → ({:.0}, {:.0})",
            self.target_pos.x, self.target_pos.y
        )
    }
}
