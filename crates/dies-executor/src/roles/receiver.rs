use std::sync::{atomic::AtomicBool, Arc};

use dies_core::{PlayerData, WorldData};
use nalgebra::Vector2;

use crate::{roles::skills::{FetchBall, GoToPosition}, skill, PlayerControlInput};

use super::{Role, RoleCtx};


pub struct Receiver {
    has_passer_kicked: bool,
}

impl Receiver {
    pub fn new() -> Self {
        Self {
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
            skill!(ctx, FetchBall::new());

            skill!(ctx, GoToPosition::new(Vector2::new(0.0, 0.0)).with_ball());
        }
        
        PlayerControlInput::default()
    }
}
