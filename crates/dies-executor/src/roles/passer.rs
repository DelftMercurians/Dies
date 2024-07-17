use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use dies_core::{Angle, PlayerId};
use dies_core::{PlayerData, WorldData};

use crate::roles::skills::{ApproachBall, Face, FetchBallWithHeading, Kick};
use crate::roles::{Role, SkillResult};
use crate::{skill, PlayerControlInput};

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
            if let SkillResult::Success = skill!(ctx, Kick::new()) {
                break;
            }
        }

        self.has_passed = true;

        PlayerControlInput::default()
    }
}
