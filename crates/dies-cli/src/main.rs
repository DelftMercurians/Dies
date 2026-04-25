use std::process::ExitCode;

use clap::Parser;
use cli::Cli;

mod cli;
mod commands;

#[tokio::main]
async fn main() -> ExitCode {
    // console_subscriber::init();

    println!("Dies CLI v{}", env!("CARGO_PKG_VERSION"));

    let res = tokio::spawn(async { Cli::parse().start().await });

    match res.await {
        Err(_) => ExitCode::FAILURE,
        Ok(code) => code,
    }
}
