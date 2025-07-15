use dies_core::{Angle, PlayerId, Vector2};

use super::{HeadingTarget, PlayerControlInput, SkillCtx, SkillProgress};

#[derive(Clone)]
pub struct Face {
    heading: HeadingTarget,
    with_ball: bool,
}

impl Face {
    pub fn new(heading: Angle) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Angle(heading),
        }
    }

    pub fn towards_position(pos: Vector2) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::Position(pos),
        }
    }

    pub fn towards_own_player(id: PlayerId) -> Self {
        Self {
            with_ball: false,
            heading: HeadingTarget::OwnPlayer(id),
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
            input.with_dribbling(0.4); // turn down dribbler when turning
            input.with_angular_acceleration_limit(240.0f64.to_radians());
            input.with_angular_speed_limit(1.0); // radians
        }

        if (ctx.player.yaw - heading).abs() < 3.0f64.to_radians() {
            return SkillProgress::success();
        }
        SkillProgress::Continue(input)
    }
}
