//! Collision-avoidance stack: a shared obstacle model read by two independent
//! layers — a global path planner ("which way") and ORCA reciprocal avoidance
//! ("don't crash"). MTP ("how fast") stays separate and dumb.

mod obstacle;
mod orca;
mod planner;

pub use obstacle::{AvoidanceGates, ObstacleSet};
pub use orca::{orca_solve_batch, OrcaAgent, OrcaSolver};
pub use planner::{GlobalPlanner, PlanStep};
