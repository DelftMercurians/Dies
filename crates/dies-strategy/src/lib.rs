mod testing;
mod v0;

use dies_executor::behavior_tree_api::GameContext;
pub use testing::testing_strategy;
pub use v0::v0_strategy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyDef {
    Testing,
    V0,
}

impl StrategyDef {
    pub fn get(self) -> fn(&mut GameContext) -> () {
        match self {
            StrategyDef::Testing => testing_strategy,
            StrategyDef::V0 => v0_strategy,
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "testing" => StrategyDef::Testing,
            "v0" => StrategyDef::V0,
            _ => panic!("Invalid strategy: {}", s),
        }
    }
}
