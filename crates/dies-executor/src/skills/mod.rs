pub mod skills;

mod face;
mod fetchball;
mod go_to_pos;
mod kick;
mod pass;
mod receive;
mod wait;

pub use face::Face;
pub use fetchball::FetchBall;
pub use go_to_pos::GoToPosition;
pub use kick::Kick;
pub use pass::Pass;
pub use receive::TryReceive;
pub use wait::Wait;

use crate::{behavior_tree::BtContext, control::PlayerControlInput};
use dies_core::{Angle, PlayerData, PlayerId, TeamData, Vector2};

pub enum Skill {
    GoToPosition(GoToPosition),
    Face(Face),
    Kick(Kick),
    Wait(Wait),
    FetchBall(FetchBall),
    Pass(Pass),
    TryReceive(TryReceive),
}

impl Skill {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        match self {
            Skill::GoToPosition(skill) => skill.update(ctx),
            Skill::Face(skill) => skill.update(ctx),
            Skill::Kick(skill) => skill.update(ctx),
            Skill::Wait(skill) => skill.update(ctx),
            Skill::FetchBall(skill) => skill.update(ctx),
            Skill::Pass(skill) => skill.update(ctx),
            Skill::TryReceive(skill) => skill.update(ctx),
        }
    }
}

#[derive(Clone)]
pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a TeamData,
    pub bt_context: BtContext,
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

#[derive(Clone)]
pub enum HeadingTarget {
    Angle(Angle),
    Ball,
    Position(Vector2),
    OwnPlayer(PlayerId),
}

impl HeadingTarget {
    fn heading(&self, ctx: &SkillCtx) -> Option<Angle> {
        match self {
            HeadingTarget::Angle(angle) => Some(*angle),
            HeadingTarget::Ball => ctx
                .world
                .ball
                .as_ref()
                .map(|ball| Angle::between_points(ctx.player.position, ball.position.xy())),
            HeadingTarget::Position(pos) => Some(Angle::between_points(ctx.player.position, *pos)),
            HeadingTarget::OwnPlayer(id) => {
                let player = ctx.world.get_player(*id);
                Some(Angle::between_points(ctx.player.position, player.position))
            }
        }
    }
}
