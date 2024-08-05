use dies_core::PlayerId;
use nalgebra::Vector2;

use super::{Role, RoleCtx, SkillResult};
use crate::{
    invoke_skill,
    roles::skills::{Face, FetchBall, Kick},
    skill, PlayerControlInput,
};

pub struct Receiver {
    passer: PlayerId,
    has_passer_kicked: bool,
}

impl Receiver {
    pub fn new(passer: PlayerId) -> Self {
        Self {
            passer,
            has_passer_kicked: false,
        }
    }

    pub fn set_passer_kicked(&mut self) {
        self.has_passer_kicked = true;
    }
}

impl Role for Receiver {
    fn update(&mut self, ctx: RoleCtx) -> PlayerControlInput {
        if self.has_passer_kicked {
            loop {
                match skill!(ctx, FetchBall::new()) {
                    crate::roles::SkillResult::Success => break,
                    _ => {}
                }

                match invoke_skill!(ctx, Face::towards_position(Vector2::new(4500.0, 0.0))) {
                    crate::roles::SkillProgress::Continue(mut input) => {
                        input.with_dribbling(1.0);
                        return input;
                    }
                    _ => {}
                }

                if let SkillResult::Success = skill!(ctx, Kick::new()) {
                    break;
                }
            }
        } else {
            skill!(ctx, Face::towards_own_player(self.passer));
        }

        PlayerControlInput::default()
    }
}
