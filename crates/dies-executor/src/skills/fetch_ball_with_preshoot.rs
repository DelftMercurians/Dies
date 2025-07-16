use dies_core::Vector2;

use crate::skills::{
    FetchBall, GoToPosition, Shoot, ShootTarget, SkillCtx, SkillProgress, SkillResult,
};

#[derive(Clone)]
pub struct FetchBallWithPreshoot {
    preshoot_position: Vector2,
    preshoot_heading: Option<dies_core::Angle>,
    shoot_target: ShootTarget,
    state: FetchBallWithPreshootState,
}

#[derive(Clone)]
enum FetchBallWithPreshootState {
    None,
    GoToPreshoot(GoToPosition),
    FetchBall(FetchBall),
    Shoot(Shoot),
}

impl FetchBallWithPreshoot {
    pub fn new(preshoot_position: Vector2, shoot_target: ShootTarget) -> Self {
        Self {
            preshoot_position,
            preshoot_heading: None,
            shoot_target,
            state: FetchBallWithPreshootState::None,
        }
    }

    pub fn with_heading(mut self, heading: dies_core::Angle) -> Self {
        self.preshoot_heading = Some(heading);
        self
    }

    pub fn state(&self) -> String {
        match &self.state {
            FetchBallWithPreshootState::None => "None".to_string(),
            FetchBallWithPreshootState::GoToPreshoot(_) => {
                format!("GoToPreshoot: {:?}", self.preshoot_position)
            }
            FetchBallWithPreshootState::FetchBall(_) => "FetchBall".to_string(),
            FetchBallWithPreshootState::Shoot(_) => format!("Shoot: {:?}", self.shoot_target),
        }
    }

    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        loop {
            match &mut self.state {
                FetchBallWithPreshootState::None => {
                    let mut go_to_position = GoToPosition::new(self.preshoot_position);
                    if let Some(heading) = self.preshoot_heading {
                        go_to_position = go_to_position.with_heading(heading);
                    }
                    go_to_position = go_to_position.avoid_ball();
                    self.state = FetchBallWithPreshootState::GoToPreshoot(go_to_position);
                }
                FetchBallWithPreshootState::GoToPreshoot(go_to_position) => {
                    let progress = go_to_position.update(ctx.clone());
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            self.state = FetchBallWithPreshootState::FetchBall(FetchBall::new());
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
                FetchBallWithPreshootState::FetchBall(fetch_ball) => {
                    let progress = fetch_ball.update(ctx.clone());
                    match progress {
                        SkillProgress::Continue(input) => {
                            return SkillProgress::Continue(input);
                        }
                        SkillProgress::Done(SkillResult::Success) => {
                            self.state = FetchBallWithPreshootState::Shoot(Shoot::new(
                                self.shoot_target.clone(),
                            ));
                        }
                        SkillProgress::Done(SkillResult::Failure) => {
                            return SkillProgress::Done(SkillResult::Failure);
                        }
                    }
                }
                FetchBallWithPreshootState::Shoot(shoot) => {
                    let progress = shoot.update(ctx.clone());
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
