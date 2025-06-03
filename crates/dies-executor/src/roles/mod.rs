pub mod skills;

use crate::control::PlayerControlInput;
use dies_core::{PlayerData, WorldData};

use skills::{Face, FetchBall, GoToPosition, Kick, Wait};

#[derive(Clone)]
pub enum Skill {
    GoToPosition(GoToPosition),
    Face(Face),
    Kick(Kick),
    Wait(Wait),
    FetchBall(FetchBall),
}

impl Skill {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        match self {
            Skill::GoToPosition(skill) => skill.update(ctx),
            Skill::Face(skill) => skill.update(ctx),
            Skill::Kick(skill) => skill.update(ctx),
            Skill::Wait(skill) => skill.update(ctx),
            Skill::FetchBall(skill) => skill.update(ctx),
        }
    }
}

pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a WorldData,
}

/// The result of a skill execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillResult {
    Success,
    Failure,
}

/// The state of a skill execution
pub enum SkillState {
    InProgress(Skill),
    Done(SkillResult),
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

    pub fn map_input<F>(self, f: F) -> SkillProgress
    where
        F: FnOnce(PlayerControlInput) -> PlayerControlInput,
    {
        match self {
            SkillProgress::Continue(input) => SkillProgress::Continue(f(input)),
            SkillProgress::Done(result) => SkillProgress::Done(result),
        }
    }
}
