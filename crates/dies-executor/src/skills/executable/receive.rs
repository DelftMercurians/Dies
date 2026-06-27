use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use crate::control::skill_executor::{ExecutableSkill, SkillContext, SkillProgress};
use crate::control::PlayerControlInput;

/// Minimum closing speed (mm/s) along the pass line before the receiver commits
/// to an interception. Below this the ball isn't meaningfully on its way, so the
/// receiver holds the planned intercept point instead of chasing it.
const MIN_CLOSING_SPEED: f64 = 250.0;
/// Maximum look-ahead (s) for the interception prediction. A real pass arrives
/// well within this; a longer predicted crossing time means the ball is barely
/// converging (or diverging), so the receiver holds rather than chases.
const MAX_PREDICT_TIME: f64 = 2.5;

#[derive(Clone)]
pub struct ReceiveSkill {
    from_pos: Vector2,
    target_pos: Vector2,
    capture_limit: f64,
    cushion: bool,
    status: SkillStatus,
}

impl ReceiveSkill {
    pub fn new(from_pos: Vector2, target_pos: Vector2, capture_limit: f64, cushion: bool) -> Self {
        Self {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
            status: SkillStatus::Running,
        }
    }

    /// The pass-line origin this receiver is intercepting from.
    pub fn from_pos(&self) -> Vector2 {
        self.from_pos
    }

    /// Update the geometry in place (used when composed inside the pass
    /// coordinator, which feeds fresh passer/intercept positions each frame).
    pub fn reconfigure(
        &mut self,
        from_pos: Vector2,
        target_pos: Vector2,
        capture_limit: f64,
        cushion: bool,
    ) {
        self.from_pos = from_pos;
        self.target_pos = target_pos;
        self.capture_limit = capture_limit;
        self.cushion = cushion;
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
        if ctx.player.has_ball {
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

        // Position to intercept the pass. The receiver holds its depth at the
        // planned intercept point and only slides laterally (along the pass-line
        // normal) to meet a ball that is genuinely on its way. If the ball is not
        // closing toward us, hold the intercept point instead of chasing the
        // ball's lateral projection across the field.
        let target_position = match ctx.world.ball.as_ref() {
            Some(ball) => {
                let ball_pos = ball.position.xy();
                let ball_vel = ball.velocity.xy();

                let line_vec = self.target_pos - self.from_pos;
                if line_vec.norm() < 1.0 {
                    // Degenerate geometry (passer ~on the intercept point): hold.
                    self.target_pos
                } else {
                    let line_dir = line_vec.normalize();
                    let normal = Vector2::new(-line_vec.y, line_vec.x).normalize();

                    // Closing speed of the ball along the pass line (passer -> target).
                    let closing_speed = ball_vel.dot(&line_dir);
                    if closing_speed > MIN_CLOSING_SPEED {
                        // Time for the ball to reach our depth (the normal line
                        // through the intercept point).
                        let to_ball = ball_pos - self.target_pos;
                        let t = -to_ball.dot(&line_dir) / closing_speed;
                        if t > 0.0 && t < MAX_PREDICT_TIME {
                            // Slide laterally to where the ball will cross our depth.
                            let future_ball_pos = ball_pos + ball_vel * t;
                            let lateral = (future_ball_pos - self.target_pos)
                                .dot(&normal)
                                .clamp(-self.capture_limit, self.capture_limit);
                            self.target_pos + normal * lateral
                        } else {
                            // Crossing is in the past or too far out in time: hold.
                            self.target_pos
                        }
                    } else {
                        // Ball is moving away or not converging: hold.
                        self.target_pos
                    }
                }
            }
            // No ball detected: hold the planned intercept point.
            None => self.target_pos,
        };

        input.with_position(target_position);

        SkillProgress::Continue(input)
    }

    fn status(&self) -> SkillStatus {
        self.status
    }

    fn skill_type(&self) -> &'static str {
        "Receive"
    }

    fn description(&self) -> String {
        format!(
            "intercepting from ({:.0}, {:.0})",
            self.from_pos.x, self.from_pos.y
        )
    }
}
