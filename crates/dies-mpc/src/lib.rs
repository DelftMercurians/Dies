//! Pure-Rust model-predictive controller for translational motion of
//! omnidirectional SSL robots.
//!
//! Minimal viable shape: first-order velocity-lag dynamics (one τ per body
//! axis), pure-quadratic stage cost (pos / vel / control / smoothness),
//! iLQR solver. Soft obstacle barriers (robots, field walls, defense areas)
//! are an optional, position-only addition to the stage cost — the geometry is
//! supplied by the integration layer via `MpcTarget::obstacles`.

pub mod cost;
pub mod dynamics;
mod generated;
pub mod obstacle;
pub mod solver;
pub mod types;

pub use obstacle::{Obstacle, ObstacleShape};
pub use solver::solve;
pub use types::*;
