use std::time::Instant;

use dies_core::SysStatus;

use super::{SkillCtx, SkillProgress};
use crate::{KickerControlInput, PlayerControlInput};

#[derive(Clone)]
pub struct Kick {
    has_kicked: usize,
    timer: Option<Instant>,
}

impl Kick {
    pub fn new() -> Self {
        Self {
            has_kicked: 1,
            timer: None,
        }
    }
}

impl Kick {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let mut input = PlayerControlInput::new();
        input.with_dribbling(1.0);

        if self.has_kicked == 0 {
            let timer = self.timer.get_or_insert(Instant::now());
            if timer.elapsed().as_secs_f64() > 0.1 {
                if ctx
                    .world
                    .ball
                    .as_ref()
                    .map(|b| b.detected && (b.position.xy() - ctx.player.position).norm() > 200.0)
                    .unwrap_or(false)
                {
                    return SkillProgress::success();
                } else {
                    return SkillProgress::failure();
                }
            } else {
                return SkillProgress::Continue(input);
            }
        }

        let ready = match ctx.player.kicker_status.as_ref() {
            Some(SysStatus::Ready) => true,
            _ => false,
        };

        if ready {
            if !ctx.player.breakbeam_ball_detected
                && (ctx
                    .world
                    .ball
                    .as_ref()
                    .map(|b| b.detected && ((b.position.xy() - ctx.player.position).norm() < 200.0))
                    .unwrap_or(false))
            {
                return SkillProgress::failure();
            }
            input.with_kicker(KickerControlInput::Kick { force: 1.0 });
            self.has_kicked -= 1;
            SkillProgress::Continue(input)
        } else {
            input.kicker = KickerControlInput::Arm;
            SkillProgress::Continue(input)
        }
    }
}
