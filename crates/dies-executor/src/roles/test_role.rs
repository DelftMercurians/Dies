use dies_core::{dbg_draw, Vector2};

use crate::{
    roles::{skills::GoToPositionSkill, Role},
    skill, PlayerControlInput,
};

use super::RoleCtx;

pub struct TestRole {}

impl Role for TestRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        dbg_draw!(("p{}.target", ctx.player.id), cross, Vector2::new(0.0, 0.0));
        skill!(ctx, GoToPositionSkill::new(Vector2::new(0.0, 0.0)));
        PlayerControlInput::new()
    }
}
