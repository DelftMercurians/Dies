use dies_core::SysStatus;

use super::{SkillCtx, SkillProgress};
use crate::{control::Velocity, KickerControlInput, PlayerControlInput};

#[derive(Clone)]
pub struct Kick {
    has_kicked: bool,
}

impl Kick {
    pub fn new() -> Self {
        Self { has_kicked: false }
    }
}

impl Kick {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();

        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_pos = ball.position.xy();
            let player_pos = ctx.player.position.xy();
            let dir = (ball_pos - player_pos).normalize();
            input.velocity = Velocity::Global(dir * 100.0);
        }

        input.with_dribbling(0.6);

        if self.has_kicked {
            return SkillProgress::Done(super::SkillResult::Success);
        }
        if !ctx.player.breakbeam_ball_detected {
            return SkillProgress::failure();
        }

        input.with_kicker(KickerControlInput::Kick { force: 1.0 });
        self.has_kicked = true;
        SkillProgress::Continue(input)
    }
}
