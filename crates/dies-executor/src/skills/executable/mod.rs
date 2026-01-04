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

mod go_to_pos;
mod dribble;
mod pickup_ball;
mod reflex_shoot;

pub use go_to_pos::GoToPosSkill;
pub use dribble::DribbleSkill;
pub use pickup_ball::PickupBallSkill;
pub use reflex_shoot::ReflexShootSkill;

