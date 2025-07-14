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

        let ready = match ctx.player.kicker_status.as_ref() {
            Some(SysStatus::Ready) => true,
            _ => false,
        };

        if ready {
            input.with_kicker(KickerControlInput::Kick { force: 1.0 });
            self.has_kicked = true;
            SkillProgress::Continue(input)
        } else {
            input.kicker = KickerControlInput::Arm;
            SkillProgress::Continue(input)
        }
    }
}
