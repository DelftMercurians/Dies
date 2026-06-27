use dies_strategy_runner::run_strategy;
use v0_strategy::V0Strategy;

fn main() {
    run_strategy(V0Strategy::new);
}
