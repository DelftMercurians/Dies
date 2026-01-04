//! # dies-strategy-runner
//!
//! Process runner for dies strategy binaries.
//!
//! This crate provides the infrastructure for running strategy processes
//! and communicating with the executor via IPC.
//!
//! # Usage
//!
//! Create a strategy binary with the following structure:
//!
//! ```ignore
//! use dies_strategy_runner::run_strategy;
//! use my_strategy::MyStrategy;
//!
//! fn main() {
//!     run_strategy(MyStrategy::new);
//! }
//! ```

pub mod ipc;
pub mod logging;
pub mod runner;

pub use runner::{run_strategy, run_strategy_with_args, RunnerArgs, StrategyRunner};
