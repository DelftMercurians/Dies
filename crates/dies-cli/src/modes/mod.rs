use clap::ValueEnum;

pub mod irl;
pub mod sim_test;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Mode {
    Irl,
    SimTest,
}
