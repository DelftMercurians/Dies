use super::RoleCtx;
use crate::{
    roles::{
        skills::{GoToPosition, Wait},
        Role,
    },
    skill, PlayerControlInput,
};
use dies_core::Vector2;

pub struct TestRole {
    targets: Vec<Vector2>,
}

impl TestRole {
    pub fn new(targets: Vec<Vector2>) -> Self {
        Self { targets }
    }
}

impl Role for TestRole {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        for target in &self.targets {
            skill!(ctx, GoToPosition::new(target.clone()));
            skill!(ctx, Wait::new_secs_f64(1.0));
        }

        PlayerControlInput::default()
    }
}
