use dies_core::{Angle, PlayerId, Vector2};

use crate::skills::SkillResult;

use super::{HeadingTarget, PlayerControlInput, SkillCtx, SkillProgress};

#[derive(Clone)]
pub struct Face {
    heading: HeadingTarget,
    with_ball: bool,
    last_err: Option<f64>,
}

impl Face {
    pub fn new(heading: Angle) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Angle(heading),
            last_err: None,
        }
    }

    pub fn towards_position(pos: Vector2) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Position(pos),
            last_err: None,
        }
    }

    pub fn towards_own_player(id: PlayerId) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::OwnPlayer(id),
            last_err: None,
        }
    }

    pub fn with_ball(mut self) -> Self {
        self.with_ball = true;
        self
    }
}

impl Face {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();
        if let Some(ball) = ctx.world.ball.as_ref() {
            let balldist = (ball.position.xy() - ctx.player.position).magnitude();
            if self.with_ball && balldist > 350.0 {
                return SkillProgress::failure();
            }
        }
        let heading = if let Some(heading) = self.heading.heading(&ctx) {
            heading
        } else {
            log::error!("No heading found for face");
            return SkillProgress::failure();
        };

        input.with_yaw(heading);
        input.with_care(0.7); // this compensates for flakiness of with_ball
        if self.with_ball {
            input.with_dribbling(0.6); // turn down dribbler when turning
            input.with_angular_acceleration_limit(240.0f64.to_radians());
            input.with_angular_speed_limit(1.3); // radians
            if !ctx.player.breakbeam_ball_detected {
                return SkillProgress::Done(SkillResult::Failure);
            }
        }

        self.last_err = Some((ctx.player.yaw - heading).abs());
        if (ctx.player.yaw - heading).abs() < 6.0f64.to_radians() {
            return SkillProgress::success();
        }
        SkillProgress::Continue(input)
    }

    pub fn get_last_err(&self) -> Option<f64> {
        self.last_err
    }
}
