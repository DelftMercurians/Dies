
use optimization_engine::{alm::*, constraints::*, core::*, panoc::*}

// define the system's state and control input dimensions
const STATE_DIM: usize = 2;
const CONTROL_DIM: usize = 2;





struct MPCParams{
    q: Vec<f64>, //State cost weight
    r: Vec<f64>, // Control input cost weight
    // add other params here
}



impl ModelPredictiveControl {}