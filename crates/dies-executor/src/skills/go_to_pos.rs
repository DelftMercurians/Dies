use dies_core::{Angle, Vector2};

use super::{PlayerControlInput, SkillCtx, SkillProgress};

const DEFAULT_POS_TOLERANCE: f64 = 50.0;
const DEFAULT_VEL_TOLERANCE: f64 = 20.0;

/// A skill that makes the player go to a specific position
#[derive(Clone)]
pub struct GoToPosition {
    target_pos: Vector2,
    target_heading: Option<Angle>,
    target_velocity: Vector2,
    pos_tolerance: f64,
    velocity_tolerance: f64,
    with_ball: bool,
    avoid_ball: bool,
}

impl GoToPosition {
    pub fn new(target_pos: Vector2) -> Self {
        Self {
            target_pos,
            target_heading: None,
            target_velocity: Vector2::zeros(),
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            velocity_tolerance: DEFAULT_VEL_TOLERANCE,
            with_ball: false,
            avoid_ball: false,
        }
    }

    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.target_heading = Some(heading);
        self
    }

    /// Drive with the ball.
    ///
    /// This activates the dribbler and makes sure the relative velocity between the
    /// player and the ball is below a certain threshold.
    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }

    pub fn avoid_ball(mut self) -> Self {
        self.avoid_ball = true;
        self
    }
}

impl GoToPosition {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let position = ctx.player.position;
        let distance = (self.target_pos - position).norm();
        let dv = (self.target_velocity - ctx.player.velocity).norm();
        dies_core::debug_string(format!("goto_distance"), distance.to_string());
        dies_core::debug_string(format!("goto_dv"), dv.to_string());

        if distance < self.pos_tolerance && dv < self.velocity_tolerance {
            dies_core::debug_string(format!("goto_distance"), format!("success"));
            dies_core::debug_string(format!("goto_dv"), format!("success"));
            return SkillProgress::success();
        }

        let mut input = PlayerControlInput::new();
        input.with_position(self.target_pos);
        if let Some(heading) = self.target_heading {
            input.with_yaw(heading);
        }

        if ctx.world.ball.as_ref().is_some() {
            if self.avoid_ball {
                input.avoid_ball = true;
                if ctx.player.breakbeam_ball_detected { // TODO: fix
                    return SkillProgress::success();
                }
            } else if self.with_ball {
                if !ctx.player.breakbeam_ball_detected {
                    return SkillProgress::failure();
                }

                input.with_dribbling(1.0);
                input.with_acceleration_limit(700.0);
                input.with_angular_acceleration_limit(180.0f64.to_radians());
                input.with_angular_speed_limit(180.0f64.to_radians());
            }

            // let ball_vel = ball.velocity.xy();
            // let relative_velocity = ball_vel - ctx.player.velocity;
            // if relative_velocity.norm() > DEFAULT_BALL_VEL_TOLERANCE {
            //     let correction = relative_velocity * BALL_VEL_CORRECTION;
            //     input.add_global_velocity(correction);
            // }
        }

        SkillProgress::Continue(input)
    }
}
