//! Skills module - contains both legacy and new executable skills.
//!
//! The `executable` submodule contains the new streamlined skills for the
//! strategy-controlled path.
//!
//! The legacy skills (only available with `legacy-strategy` feature) use
//! the behavior tree context.

pub mod executable;

// Legacy skill modules (only with legacy-strategy feature)
#[cfg(feature = "legacy-strategy")]
pub mod skills;
#[cfg(feature = "legacy-strategy")]
mod face;
#[cfg(feature = "legacy-strategy")]
mod fetch_ball_with_preshoot;
#[cfg(feature = "legacy-strategy")]
mod fetchball;
#[cfg(feature = "legacy-strategy")]
mod go_to_pos;
#[cfg(feature = "legacy-strategy")]
mod kick;
#[cfg(feature = "legacy-strategy")]
mod receive;
#[cfg(feature = "legacy-strategy")]
mod shoot;
#[cfg(feature = "legacy-strategy")]
mod test_movement;
#[cfg(feature = "legacy-strategy")]
mod wait;

#[cfg(feature = "legacy-strategy")]
pub use face::Face;
#[cfg(feature = "legacy-strategy")]
pub use fetch_ball_with_preshoot::FetchBallWithPreshoot;
#[cfg(feature = "legacy-strategy")]
pub use fetchball::FetchBall;
#[cfg(feature = "legacy-strategy")]
pub use go_to_pos::GoToPosition;
#[cfg(feature = "legacy-strategy")]
pub use kick::Kick;
#[cfg(feature = "legacy-strategy")]
pub use receive::TryReceive;
#[cfg(feature = "legacy-strategy")]
pub use shoot::Shoot;
#[cfg(feature = "legacy-strategy")]
pub use test_movement::TestMovement;
#[cfg(feature = "legacy-strategy")]
pub use wait::Wait;

#[cfg(feature = "legacy-strategy")]
use crate::{
    behavior_tree::BtContext,
    control::{PlayerContext, PlayerControlInput},
    TeamContext,
};
#[cfg(feature = "legacy-strategy")]
use dies_core::{Angle, PlayerData, PlayerId, TeamData, Vector2};

#[cfg(feature = "legacy-strategy")]
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

#[cfg(feature = "legacy-strategy")]
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

#[cfg(feature = "legacy-strategy")]
#[derive(Clone)]
pub struct SkillCtx<'a> {
    pub player: &'a PlayerData,
    pub world: &'a TeamData,
    pub bt_context: BtContext,
    pub viz_path_prefix: String,
    pub team_context: TeamContext,
}

#[cfg(feature = "legacy-strategy")]
/// The result of a skill execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillResult {
    Success,
    Failure,
}

#[cfg(feature = "legacy-strategy")]
/// The progress of a skill execution
#[derive(Debug)]
pub enum SkillProgress {
    Continue(PlayerControlInput),
    Done(SkillResult),
}

#[cfg(feature = "legacy-strategy")]
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

#[cfg(feature = "legacy-strategy")]
#[derive(Clone)]
pub enum HeadingTarget {
    Angle(Angle),
    Ball,
    Position(Vector2),
    OwnPlayer(PlayerId),
}

#[cfg(feature = "legacy-strategy")]
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
