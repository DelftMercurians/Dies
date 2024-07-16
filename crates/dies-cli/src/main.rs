use cli::Cli;
use std::process::ExitCode;

mod cli;
mod commands;

#[tokio::main]
async fn main() -> ExitCode {
    console_subscriber::init();

    println!("Dies CLI v{}", env!("CARGO_PKG_VERSION"));

    Cli::start().await
}
