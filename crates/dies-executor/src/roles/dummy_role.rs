use std::time::Instant;

use super::{RoleCtx, Skill, SkillCtx,SkillProgress, SkillResult};
use crate::{roles::Role, PlayerControlInput};
use dies_core::Vector2;

pub struct DummyRole {
    skill: Box<dyn Skill>
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
        return match self.skill.update(SkillCtx{
                player,
                world
          }) {
            SkillProgress::Continue(control) => control,
            _ => PlayerControlInput::default()
        }
          
    }
}
