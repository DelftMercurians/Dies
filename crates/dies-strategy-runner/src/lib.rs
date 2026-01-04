//! # dies-strategy-runner
//!
//! Process runner for dies strategy binaries.
//!
//! This crate provides the infrastructure for running strategy processes
//! and communicating with the executor via IPC.

pub mod ipc;
pub mod logging;
pub mod runner;

pub use runner::StrategyRunner;
