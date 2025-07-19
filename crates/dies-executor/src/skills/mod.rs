pub mod skills;

mod face;
mod fetch_ball_with_preshoot;
mod fetchball;
mod go_to_pos;
mod kick;
mod receive;
mod shoot;
mod test_movement;
mod wait;

pub use face::Face;
pub use fetch_ball_with_preshoot::FetchBallWithPreshoot;
pub use fetchball::FetchBall;
pub use go_to_pos::GoToPosition;
pub use kick::Kick;
pub use receive::TryReceive;
pub use shoot::Shoot;
pub use test_movement::TestMovement;
pub use wait::Wait;

use crate::{
    behavior_tree::BtContext,
    control::{PlayerContext, PlayerControlInput},
    TeamContext,
};
use dies_core::{Angle, PlayerData, PlayerId, TeamData, Vector2};

pub enum Skill {
    GoToPosition(GoToPosition),
    Face(Face),
    Kick(Kick),
    Wait(Wait),
    FetchBall(FetchBall),
    FetchBallWithPreshoot(FetchBallWithPreshoot),
    Shoot(Shoot),
    TryReceive(TryReceive),
    TestMovement(TestMovement),
}

impl Skill {
    pub fn update(&mut self, ctx: SkillCtx<'_>) -> SkillProgress {
        match self {
            Skill::GoToPosition(skill) => skill.update(ctx),
            Skill::Face(skill) => skill.update(ctx),
            Skill::Kick(skill) => skill.update(ctx),
            Skill::Wait(skill) => skill.update(ctx),
            Skill::FetchBall(skill) => skill.update(ctx),
            Skill::FetchBallWithPreshoot(skill) => skill.update(ctx),
            Skill::Shoot(skill) => skill.update(ctx),
            Skill::TryReceive(skill) => skill.update(ctx),
            Skill::TestMovement(skill) => skill.update(ctx),
        }
    }
}

#[derive(Clone)]
pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a TeamData,
    pub bt_context: BtContext,
    pub viz_path_prefix: String,
    pub team_context: TeamContext,
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
