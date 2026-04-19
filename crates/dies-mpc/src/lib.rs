//! Pure-Rust model-predictive controller for translational motion of
//! omnidirectional SSL robots.
//!
//! No external optimization or automatic-differentiation dependencies:
//! iLQR and Levenberg-Marquardt are hand-rolled on top of small dense
//! `nalgebra` matrices.
//!
//! The crate is intentionally independent of `dies-core` and the rest
//! of the workspace. Callers convert their world types into the simple
//! structs in [`types`] at the integration boundary.

pub mod barrier;
pub mod cost;
pub mod dynamics;
pub mod solver;
pub mod sysid;
pub mod types;

pub use solver::solve;
pub use sysid::fit_params;
pub use types::*;
