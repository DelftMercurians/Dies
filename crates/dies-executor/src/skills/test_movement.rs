use std::time::Instant;

use dies_core::Vector2;

use crate::{
    skills::{SkillCtx, SkillProgress},
    ControlParameters, PlayerControlInput,
};

enum Stage {
    IncreasingThresh,
    IncreasingKp,
    IncreasingKi,
}

const NUM_STEPS: usize = 10;

pub struct TestMovement {
    target1: Vector2,
    target2: Vector2,
    going_to_target1: bool,
    waiting: bool,
    timer_start: Option<Instant>,

    stage: Stage,
    step: usize,

    current_kp: f64,
    current_ki: f64,
    current_thresh: f64,
    antiwindup: f64,
}

impl TestMovement {
    pub fn new(target1: Vector2, target2: Vector2) -> Self {
        Self {
            target1,
            target2,
            waiting: false,
            timer_start: None,
            going_to_target1: true,
            current_kp: 1.0,
            current_ki: 0.001,
            current_thresh: 100.0,
            antiwindup: 40.0,
            stage: Stage::IncreasingKp,
            step: 0,
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        let timer = self.timer_start.get_or_insert(Instant::now()).clone();
        if self.waiting {
            if timer.elapsed().as_secs_f64() > 2.0 {
                self.waiting = false;
                self.timer_start = Some(Instant::now());
            }
        }
        let target = if self.going_to_target1 {
            self.target1
        } else {
            self.target2
        };
        let cutoff = 20.0;
        let error = (target - ctx.player.position).norm();
        let player_ctx = ctx.team_context.player_context(ctx.player.id);
        player_ctx.debug_value("error", error);

        if error < cutoff || timer.elapsed().as_secs_f64() > 5.0 {
            self.going_to_target1 = !self.going_to_target1;
            self.timer_start = Some(Instant::now());
            self.waiting = true;
            // Move to next stage
            if self.step < NUM_STEPS {
                self.step += 1;
            } else {
                let new_stage = match self.stage {
                    Stage::IncreasingThresh => {
                        self.current_thresh *= 1.2;
                        Stage::IncreasingKp
                    }
                    Stage::IncreasingKp => {
                        self.current_kp *= 1.3;
                        Stage::IncreasingKi
                    }
                    Stage::IncreasingKi => {
                        self.current_ki *= 1.3;
                        Stage::IncreasingThresh
                    }
                };
                self.stage = new_stage;
                self.step = 0;
            }
        }

        let mut input = PlayerControlInput::new();
        input.with_position(target);
        input.control_paramer_override = Some(ControlParameters {
            kp: self.current_kp,
            ki: self.current_ki,
            thresh: self.current_thresh,
            antiwindup: self.antiwindup,
        });

        return SkillProgress::Continue(input);
    }
}
