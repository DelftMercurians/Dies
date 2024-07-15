use clap::Parser;
use cli_modes::CliMode;
use dies_logger::AsyncProtobufLogger;
use log::{LevelFilter, Log};
use std::{process::ExitCode, str::FromStr};
use tokio::sync::broadcast;

mod cli_modes;
mod convert_logs;
mod tui_utils;

#[tokio::main]
async fn main() -> ExitCode {
    println!("Dies CLI v{}", env!("CARGO_PKG_VERSION"));

    console_subscriber::init();

    let args = tui_utils::CliArgs::parse();

    // Set up logging
    let log_dir_path = match args.ensure_log_dir_path().await {
        Ok(path) => path,
        Err(err) => {
            log::error!("Failed to create log directory: {}", err);
            return ExitCode::FAILURE;
        }
    };
    println!("Saving logs to {}", log_dir_path.display());
    let stdout_env = env_logger::Builder::new()
        .filter_level(LevelFilter::from_str(&args.log_level).expect("Invalid log level"))
        .format_timestamp(None)
        .format_module_path(false)
        .build();
    let logger = AsyncProtobufLogger::init_with_env_logger(log_dir_path.clone(), stdout_env);
    log::set_logger(logger).unwrap(); // Safe to unwrap because we know no logger has been set yet
    log::set_max_level(log::LevelFilter::Debug);
    log::info!("Saving logs to {}", log_dir_path.display());

    let (stop_tx, stop_rx) = broadcast::channel(1);
    let main_task = tokio::spawn(async move {
        if let Err(err) = CliMode::run(args, stop_rx).await {
            log::error!("Error in main task: {}", err);
        }
    });

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    logger.flush();
    println!("Shutting down (timeout 3 seconds)... Press ctrl-c again to force shutdown");
    // Allow the logger to flush before shutting down
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Fool-proof timeout for shutdown
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(3));
        eprintln!("Shutdown timed out");
        std::process::exit(1);
    });

    let shutdown_fut = async move {
        stop_tx.send(()).expect("Failed to send stop signal");
        let _ = main_task.await.expect("Executor task failed");
    };
    tokio::select! {
        _ = shutdown_fut => {}
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Forced shutdown");
            std::process::exit(1);
        }
    };

    ExitCode::SUCCESS
}
