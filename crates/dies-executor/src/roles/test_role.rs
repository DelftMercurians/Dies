use std::{borrow::Cow, vec};

use dies_core::{dbg_draw, Vector2};
use dodgy_2d::{Agent, AvoidanceOptions, Obstacle, Vec2};

use crate::{
    control::Velocity,
    roles::{skills::GoToPositionSkill, Role, SkillResult},
    skill, PlayerControlInput,
};

use super::{skills, RoleCtx};

pub struct TestRole {
    target: Vector2,
}

impl TestRole {
    pub fn new(target: Vector2) -> Self {
        Self { target }
    }
}

impl Role for TestRole {
    fn update(&mut self, _ctx: RoleCtx<'_>) -> PlayerControlInput {
        let mut input = PlayerControlInput::default();
        input.with_position(self.target);
        input
    }
}
