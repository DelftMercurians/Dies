//! Strategy Host - manages external strategy processes.
//!
//! The Strategy Host is responsible for:
//! - Spawning and managing strategy processes
//! - IPC communication via Unix domain sockets
//! - Coordinate transformation between world and team-relative frames
//! - Forwarding skill commands from strategies to the executor
//! - Forwarding debug data from strategies to the UI
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      StrategyHost                           │
//! │  ┌─────────────────┐         ┌─────────────────┐            │
//! │  │ StrategyConn    │         │ StrategyConn    │ (per-team) │
//! │  │ (Blue Team)     │         │ (Yellow Team)   │            │
//! │  └────────┬────────┘         └────────┬────────┘            │
//! │           │ Unix Socket               │ Unix Socket         │
//! └───────────┼───────────────────────────┼─────────────────────┘
//!             │                           │
//!             ▼                           ▼
//!     ┌───────────────┐           ┌───────────────┐
//!     │ Strategy      │           │ Strategy      │
//!     │ Process (Blue)│           │ Process (Yel) │
//!     └───────────────┘           └───────────────┘
//! ```

mod connection;
mod host;
mod transform;

pub use connection::{ConnectionError, StrategyConnection};
pub use host::{StrategyHost, StrategyHostConfig};
pub use transform::CoordinateTransformer;

