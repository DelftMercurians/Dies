use dies_core::{PlayerId, Vector2};

use crate::{
    behavior_tree::PassingTarget,
    skills::{Face, FetchBall, Kick, SkillCtx, SkillProgress, SkillResult},
};

enum PassState {
    FetchBall(FetchBall),
    Face(Face),
    Kick(Kick),
}

pub struct Pass {
    target_player: PlayerId,
    target_position: Vector2,
    state: PassState,
}

impl Pass {
    pub fn new(target_player: PlayerId, target_position: Vector2) -> Self {
        Self {
            target_player,
            target_position,
            state: PassState::FetchBall(FetchBall::new()),
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        loop {
            match &mut self.state {
                PassState::FetchBall(skill) => {
                    let progress = skill.update(ctx.clone());
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            self.state =
                                PassState::Face(Face::towards_position(self.target_position));
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
                PassState::Face(face) => {
                    let progress = face.update(ctx.clone());
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            self.state = PassState::Kick(Kick::new());
                            ctx.bt_context.set_passing_target(PassingTarget {
                                id: self.target_player,
                                position: self.target_position,
                            });
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
                PassState::Kick(kick) => {
                    let progress = kick.update(ctx.clone());
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            return SkillProgress::Done(SkillResult::Success);
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
            }
        }
    }
}
