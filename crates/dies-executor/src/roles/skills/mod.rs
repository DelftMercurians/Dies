use dies_core::Vector2;

use crate::PlayerControlInput;

use super::{Skill, SkillCtx, SkillResult};

const DEFAULT_TOLERANCE: f64 = 70.0;

/// A skill that makes the player go to a specific position
pub struct GoToPositionSkill {
    target: Vector2,
    tolerance: f64,
}

impl GoToPositionSkill {
    pub fn new(target: Vector2) -> Self {
        Self {
            target,
            tolerance: DEFAULT_TOLERANCE,
        }
    }
}

impl Skill for GoToPositionSkill {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillResult {
        let position = ctx.player.position;
        let distance = (self.target - position).norm();
        if distance < self.tolerance {
            return SkillResult::Done;
        }
        let mut input = PlayerControlInput::new();
        input.with_position(self.target);
        SkillResult::Continue(input)
    }
}
