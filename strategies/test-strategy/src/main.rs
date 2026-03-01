//! Test Strategy Binary
//!
//! This is the entry point for the test-strategy binary.
//! It uses the dies-strategy-runner to handle IPC and the main loop.

use dies_strategy_runner::run_strategy;
use test_strategy::TestStrategy;

fn main() {
    run_strategy(TestStrategy::new);
}
