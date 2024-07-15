use super::RoleCtx;
use crate::{
    roles::Role,
    KickerControlInput, PlayerControlInput
};

pub struct KickerRole {
}

impl KickerRole {
    pub fn new() -> Self {
     Self {}
    }
}

impl Role for KickerRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();

        input.with_kicker(KickerControlInput::Kick).clone()
    }
}
