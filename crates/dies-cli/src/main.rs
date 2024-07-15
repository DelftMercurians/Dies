use std::process::ExitCode;
use cli::Cli;

mod commands;
mod cli;

#[tokio::main]
async fn main() -> ExitCode {
    println!("Dies CLI v{}", env!("CARGO_PKG_VERSION"));

    Cli::start().await
}
