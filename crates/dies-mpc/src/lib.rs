//! Pure-Rust model-predictive controller for translational motion of
//! omnidirectional SSL robots.
//!
//! Minimal viable shape: first-order velocity-lag dynamics (one τ per body
//! axis), pure-quadratic cost (pos / vel / control / smoothness / terminal
//! pos / terminal vel), iLQR solver. No obstacles, no field bounds — those
//! belong above this crate, in the integration layer.

pub mod cost;
pub mod dynamics;
pub mod solver;
pub mod types;

pub use solver::solve;
pub use types::*;
