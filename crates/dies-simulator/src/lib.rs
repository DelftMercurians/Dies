//! Deterministic robot-soccer physics and automatic refereeing.
//! `Simulation` preserves the crate's original public API while private modules
//! own physics, robot hardware, ball interactions, referee state, and outputs.
mod ball;
mod builder;
mod config;
mod packets;
mod physics;
mod realism;
mod referee;
mod robot;
mod simulation;
mod time;

pub use builder::SimulationBuilder;
pub use config::SimulationConfig;
pub use realism::{RealismConfig, RealismTeams};
pub use referee::{RefereeMessage, SimRefereeEvent, SimulationGameState};
pub use simulation::{Simulation, SimulationPlayerState, SimulationState};
pub use time::Timer;
