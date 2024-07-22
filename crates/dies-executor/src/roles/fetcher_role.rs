use dies_core::{Angle, Vector2};

use super::RoleCtx;
use crate::{
    invoke_skill,
    roles::{
        skills::{FetchBall, GoToPosition, Kick},
        Role, SkillProgress, SkillResult,
    },
    skill, PlayerControlInput,
};

pub struct FetcherRole {}

impl FetcherRole {
    pub fn new() -> Self {
        Self {}
    }
}

impl Role for FetcherRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        loop {
            match skill!(ctx, FetchBall::new()) {
                crate::roles::SkillResult::Success => {}
                crate::roles::SkillResult::Failure => continue,
            }

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
                SkillProgress::Done(SkillResult::Failure) => continue,
                SkillProgress::Done(SkillResult::Success) => break,
            }
        }

        skill!(ctx, Kick::new());

        PlayerControlInput::new()
    }
}
