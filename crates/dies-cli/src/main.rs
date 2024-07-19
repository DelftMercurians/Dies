use cli::Cli;
use std::process::ExitCode;

mod cli;
mod commands;

#[tokio::main]
async fn main() -> ExitCode {
    console_subscriber::init();

    println!("Dies CLI v{}", env!("CARGO_PKG_VERSION"));

    let res = tokio::spawn(async {
        Cli::start().await
    });

    match res.await {
        Err(_) => ExitCode::FAILURE,
        Ok(code) => code,
    }
}
