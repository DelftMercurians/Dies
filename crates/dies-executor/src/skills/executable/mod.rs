//! Executable skills for the strategy-controlled path.
//!
//! This module contains streamlined skill implementations that work with
//! the [`SkillExecutor`](crate::control::skill_executor::SkillExecutor).
//!
//! These skills implement the [`ExecutableSkill`](crate::control::skill_executor::ExecutableSkill)
//! trait and support:
//! - Parameter updates while running
//! - Status reporting
//! - Clean completion semantics

mod dribble;
mod go_to_bounded;
mod go_to_pos;
mod handle_ball;
mod receive;
mod shoot;
mod snatch;

pub use dribble::DribbleSkill;
pub use go_to_bounded::GoToBoundedSkill;
pub use go_to_pos::GoToPosSkill;
pub use handle_ball::HandleBallSkill;
pub use receive::ReceiveSkill;
pub use shoot::ShootSkill;
pub use snatch::SnatchSkill;
