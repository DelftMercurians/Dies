use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct StrategyInput {
    pub worldData: WorldData,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StrategyOutput {
    pub message: TeamCmd,
}

struct TeamCmd {
    
}

/*
sharable crate: "commons"
wrap my recv in a 

write test in strategy_instance.rs
*/