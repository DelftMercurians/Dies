use super::RoleCtx;
use crate::{
    roles::{
        skills::{GoToPosition, FetchBall, Wait},
        Role,
    },
    skill, PlayerControlInput,
};
use dies_core::Vector2;
use dies_protos::ssl_gc_api::Input;

pub struct FetcherRole {
}

impl FetcherRole {
    pub fn new() -> Self {
     Self {}
    }
}

impl Role for FetcherRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        skill!(ctx, Wait::new_secs_f64(0.5));
        skill!(ctx, FetchBall::new());

        PlayerControlInput::new().with_dribbling(FetchBall::new().dribbling_distance).clone()
    }
}
