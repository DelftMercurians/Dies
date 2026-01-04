//! # dies-strategy-protocol
//!
//! This crate defines the IPC protocol and shared types used for communication
//! between the executor (host) and strategy processes.
//!
//! ## Overview
//!
//! The strategy system uses a process-based architecture where strategies run
//! in separate processes and communicate with the executor via Unix domain sockets.
//! This crate provides the message types for that communication.
//!
//! ## Key Types
//!
//! - [`HostMessage`]: Messages sent from the executor to strategy processes
//! - [`StrategyMessage`]: Messages sent from strategy processes to the executor
//! - [`WorldSnapshot`]: Complete world state sent to strategies (in normalized coordinates)
//! - [`SkillCommand`]: Commands from strategies to control player skills
//! - [`SkillStatus`]: Current status of a skill execution
//!
//! ## Coordinate System
//!
//! All coordinates in this protocol are in a **team-relative frame**:
//! - **+x axis**: Points toward the opponent's goal (attacking direction)
//! - **-x axis**: Points toward our own goal (defending direction)
//! - **y axis**: Left/right from the team's perspective
//!
//! Strategies never see absolute world coordinates or team color - the executor
//! handles all coordinate transformations.

mod debug;
mod messages;
mod skill;
mod world;

pub use debug::*;
pub use messages::*;
pub use skill::*;
pub use world::*;

// Re-export commonly used types from dies-core for convenience
pub use dies_core::{Angle, FieldGeometry, PlayerId, Vector2, Vector3};

