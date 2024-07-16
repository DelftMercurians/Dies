use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;

use dies_core::{Angle, PlayerId};
use dies_core::{PlayerData, WorldData};

use crate::roles::skills::{FetchBallWithHeading, Kick};
use crate::roles::Role;
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
        let heading = Angle::between_points(ctx.player.position, receiver.position);

        skill!(ctx, FetchBallWithHeading::new(heading));
        skill!(ctx, Kick::new());

        self.has_passed = true;

        PlayerControlInput::default()
    }
}
