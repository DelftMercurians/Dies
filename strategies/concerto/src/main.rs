use concerto::ConcertoStrategy;
use dies_strategy_runner::run_strategy;

fn main() {
    run_strategy(ConcertoStrategy::new);
}
