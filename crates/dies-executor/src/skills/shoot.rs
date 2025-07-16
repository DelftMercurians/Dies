use dies_core::{PlayerId, Vector2};

use crate::{
    behavior_tree::PassingTarget,
    skills::{Face, Kick, SkillCtx, SkillProgress, SkillResult},
};

#[derive(Clone, Debug)]
pub enum ShootTarget {
    Goal(Vector2),
    Player {
        id: PlayerId,
        position: Option<Vector2>,
    },
}

impl ShootTarget {
    pub fn position(&self) -> Option<Vector2> {
        match self {
            ShootTarget::Goal(position) => Some(*position),
            ShootTarget::Player { position, .. } => *position,
        }
    }
}

enum ShootState {
    None,
    Face(Face),
    Kick(Kick),
}

pub struct Shoot {
    target: ShootTarget,
    state: ShootState,
    has_set_flag: bool,
}

impl Shoot {
    pub fn new(target: ShootTarget) -> Self {
        Self {
            state: ShootState::None,
            target,
            has_set_flag: false,
        }
    }

    pub fn state(&self) -> String {
        match &self.state {
            ShootState::None => "None".to_string(),
            ShootState::Face(_) => format!("Face: {:?}", self.target),
            ShootState::Kick(_) => format!("Kick: {:?}", self.target),
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        loop {
            match &mut self.state {
                ShootState::None => {
                    self.state = match &self.target {
                        ShootTarget::Goal(position) => {
                            ShootState::Face(Face::towards_position(*position).with_ball())
                        }
                        ShootTarget::Player { id, position } => {
                            let position = position.unwrap_or(ctx.world.get_player(*id).position);

                            ShootState::Face(Face::towards_position(position).with_ball())
                        }
                    };
                }
                ShootState::Face(face) => {
                    let progress = face.update(ctx.clone());
                    // if face.get_last_err().unwrap_or(0.0) < 30.0f64.to_radians()
                    //     && !self.has_set_flag
                    // {
                    //     if let ShootTarget::Player { id, position } = &self.target {
                    //         ctx.bt_context.set_passing_target(PassingTarget {
                    //             id: *id,
                    //             position: position.unwrap_or(ctx.world.get_player(*id).position),
                    //         });
                    //         self.has_set_flag = true;
                    //     }
                    // }
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            if let ShootTarget::Player { id, position } = &self.target {
                                ctx.bt_context.set_passing_target(PassingTarget {
                                    id: *id,
                                    position: position
                                        .unwrap_or(ctx.world.get_player(*id).position),
                                });
                                self.has_set_flag = true;
                            }
                            self.state = ShootState::Kick(Kick::new());
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
                ShootState::Kick(kick) => {
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
