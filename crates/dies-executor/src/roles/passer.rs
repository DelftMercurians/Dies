use dies_core::PlayerId;

use crate::roles::skills::{ApproachBall, Face, FetchBallWithHeading, Kick};
use crate::roles::{Role, SkillResult};
use crate::{invoke_skill, skill, PlayerControlInput};

use super::RoleCtx;

pub struct Passer {
    has_passed: bool,
    receiver_id: PlayerId,
}

impl Passer {
    pub fn new(receiver_id: PlayerId) -> Self {
        Self {
            has_passed: false,
            receiver_id,
        }
    }

    pub fn has_passed(&self) -> bool {
        self.has_passed
    }
}

impl Role for Passer {
    fn update(&mut self, ctx: RoleCtx) -> PlayerControlInput {
        let receiver = ctx.world.get_player(self.receiver_id).unwrap();

        skill!(ctx, FetchBallWithHeading::towards_own_player(receiver.id));

        loop {
            skill!(ctx, ApproachBall::new());
            match invoke_skill!(ctx, Face::towards_own_player(receiver.id).with_ball()) {
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

        self.has_passed = true;

        PlayerControlInput::default()
    }
}
