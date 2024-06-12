use clap::ValueEnum;

pub mod live;
pub mod sim;
mod utils;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum Mode {
    Live,
    Sim,
}
