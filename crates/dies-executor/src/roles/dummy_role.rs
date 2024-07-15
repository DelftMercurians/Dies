use super::{RoleCtx, Skill, SkillCtx, SkillProgress};
use crate::{roles::Role, PlayerControlInput};

pub struct DummyRole {
    skill: Box<dyn Skill>,
}

impl DummyRole {
    pub fn new(target_skill: Box<dyn Skill>) -> Self {
        Self {
            skill: target_skill,
        }
    }
}

impl Role for DummyRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let player = ctx.player;
        let world = ctx.world;
        match self.skill.update(SkillCtx { player, world }) {
            SkillProgress::Continue(control) => control,
            _ => PlayerControlInput::default(),
        }
    }
}
