use super::RoleCtx;
use crate::{roles::Role, PlayerControlInput};
use dies_core::Vector2;

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
