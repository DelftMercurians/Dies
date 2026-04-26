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
mod go_to_pos;
mod pickup_ball;
mod receive;
mod shoot;

pub use dribble::DribbleSkill;
pub use go_to_pos::GoToPosSkill;
pub use pickup_ball::PickupBallSkill;
pub use receive::ReceiveSkill;
pub use shoot::ShootSkill;
