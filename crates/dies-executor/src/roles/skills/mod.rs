use std::time::Duration;

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

/// A skill that just waits for a certain amount of time
pub struct WaitSkill {
    amount: f64,
    until: Option<f64>,
}

impl WaitSkill {
    /// Creates a new `WaitSkill` that waits for the given amount of time (starting
    /// from the next frame)
    pub fn new(amount: Duration) -> Self {
        Self {
            amount: amount.as_secs_f64(),
            until: None,
        }
    }

    /// Creates a new `WaitSkill` that waits for the given amount of time in seconds
    /// (starting from the next frame)
    pub fn new_secs_f64(amount: f64) -> Self {
        Self {
            amount,
            until: None,
        }
    }
}

impl Skill for WaitSkill {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillResult {
        let until = *self
            .until
            .get_or_insert(ctx.world.t_received + self.amount);
        if ctx.world.t_received >= until {
            SkillResult::Done
        } else {
            SkillResult::Continue(PlayerControlInput::new())
        }
    }
}
