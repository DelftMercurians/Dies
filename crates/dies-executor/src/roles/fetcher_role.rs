use super::RoleCtx;
use crate::{
    invoke_skill, roles::{
        skills::{FetchBall, GoToPosition, Kick},
        Role, SkillProgress,
    }, skill, PlayerControlInput
};
use dies_core::{Angle, Vector2};


pub struct FetcherRole {}

impl FetcherRole {
    pub fn new() -> Self {
        Self {}
    }
}

impl Role for FetcherRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        skill!(ctx, FetchBall::new());

        match invoke_skill!(
            ctx,
            GoToPosition::new(Vector2::new(0.0, 0.0))
                .with_heading(Angle::from_degrees(0.0))
                .with_ball()
        ) {
            SkillProgress::Continue(mut input) => {
                input.with_kicker(crate::KickerControlInput::Arm);
                return input;
            }
            SkillProgress::Done(_) => {}
        }

        skill!(ctx, Kick::new());

        PlayerControlInput::new()
    }
}
