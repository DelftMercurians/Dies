use clap::ValueEnum;

pub mod irl;
pub mod irl_test;
pub mod sim;
pub mod sim_test;
mod utils;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Mode {
    Irl,
    IrlTest,
    Sim,
    SimTest,
}
