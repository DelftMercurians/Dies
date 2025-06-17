pub mod skills;

use crate::control::PlayerControlInput;
use dies_core::{PlayerData, TeamData};

use skills::{
    ApproachBall, Face, FetchBall, FetchBallWithHeading, GoToPosition, InterceptBall, Kick, Shoot,
    Wait,
};

#[derive(Clone)]
pub enum Skill {
    GoToPosition(GoToPosition),
    Face(Face),
    Kick(Kick),
    Shoot(Shoot),
    Wait(Wait),
    FetchBall(FetchBall),
    InterceptBall(InterceptBall),
    ApproachBall(ApproachBall),
    FetchBallWithHeading(FetchBallWithHeading),
}

impl Skill {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        match self {
            Skill::GoToPosition(skill) => skill.update(ctx),
            Skill::Face(skill) => skill.update(ctx),
            Skill::Kick(skill) => skill.update(ctx),
            Skill::Shoot(skill) => skill.update(ctx),
            Skill::Wait(skill) => skill.update(ctx),
            Skill::FetchBall(skill) => skill.update(ctx),
            Skill::InterceptBall(skill) => skill.update(ctx),
            Skill::ApproachBall(skill) => skill.update(ctx),
            Skill::FetchBallWithHeading(skill) => skill.update(ctx),
        }
    }
}

pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a TeamData,
}

/// The result of a skill execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillResult {
    Success,
    Failure,
}

/// The progress of a skill execution
#[derive(Debug)]
pub enum SkillProgress {
    Continue(PlayerControlInput),
    Done(SkillResult),
}

impl SkillProgress {
    /// Creates a new `SkillProgress` with a `Success` result
    pub fn success() -> SkillProgress {
        SkillProgress::Done(SkillResult::Success)
    }

    /// Creates a new `SkillProgress` with a `Failure` result
    pub fn failure() -> SkillProgress {
        SkillProgress::Done(SkillResult::Failure)
    }
}
