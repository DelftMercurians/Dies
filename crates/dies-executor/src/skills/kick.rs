use dies_core::SysStatus;

use super::{SkillCtx, SkillProgress};
use crate::{KickerControlInput, PlayerControlInput};

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
        input.with_dribbling(1.0);

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
