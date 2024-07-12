use std::time::Duration;

use dies_core::Vector2;

use crate::PlayerControlInput;

use super::{Skill, SkillCtx, SkillProgress};

const DEFAULT_POS_TOLERANCE: f64 = 70.0;
const DEFAULT_VEL_TOLERANCE: f64 = 10.0;

/// A skill that makes the player go to a specific position
pub struct GoToPositionSkill {
    target_pos: Vector2,
    target_velocity: Vector2,
    pos_tolerance: f64,
    velocity_tolerance: f64,
}

impl GoToPositionSkill {
    pub fn new(target: Vector2) -> Self {
        Self {
            target_pos: target,
            target_velocity: Vector2::zeros(),
            pos_tolerance: DEFAULT_POS_TOLERANCE,
            velocity_tolerance: DEFAULT_VEL_TOLERANCE,
        }
    }
}

impl Skill for GoToPositionSkill {
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let position = ctx.player.position;
        let distance = (self.target_pos - position).norm();
        let dv = (self.target_velocity - ctx.player.velocity).norm();
        if distance < self.pos_tolerance && dv < self.velocity_tolerance {
            return SkillProgress::success();
        }
        let mut input = PlayerControlInput::new();
        input.with_position(self.target_pos);
        input.with_global_velocity(self.target_velocity);
        SkillProgress::Continue(input)
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
    fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let until = *self.until.get_or_insert(ctx.world.t_received + self.amount);
        if ctx.world.t_received >= until {
            SkillProgress::success()
        } else {
            SkillProgress::Continue(PlayerControlInput::new())
        }
    }
}
