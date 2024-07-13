use super::RoleCtx;
use crate::{
    roles::{
        skills::{GoToPosition, FetchBall, Wait},
        Role,
    },
    skill, PlayerControlInput,
};
use dies_core::Vector2;

pub struct FetcherRole {
}

impl FetcherRole {
    pub fn new() -> Self {
     Self {}
    }
}

impl Role for FetcherRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        skill!(ctx, FetchBall::new());

        PlayerControlInput::default()
    }
}
