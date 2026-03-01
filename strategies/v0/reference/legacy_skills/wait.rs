use super::{SkillCtx, SkillProgress};
use crate::PlayerControlInput;

/// A skill that just waits for a certain amount of time
#[derive(Clone)]
pub struct Wait {
    amount: f64,
    until: Option<f64>,
}

impl Wait {
    /// Creates a new `WaitSkill` that waits for the given amount of time in seconds
    /// (starting from the next frame)
    pub fn new_secs_f64(amount: f64) -> Self {
        Self {
            amount,
            until: None,
        }
    }
}

impl Wait {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let until = *self.until.get_or_insert(ctx.world.t_received + self.amount);
        if ctx.world.t_received >= until {
            SkillProgress::success()
        } else {
            SkillProgress::Continue(PlayerControlInput::new())
        }
    }
}
