use clap::Parser;
use dies_logger::AsyncProtobufLogger;
use log::{LevelFilter, Log};
use modes::Mode;
use std::str::FromStr;
use tokio::sync::broadcast;

mod modes;
mod tui_utils;

#[tokio::main]
async fn main() {
    let args = tui_utils::CliArgs::parse();

    // Set up logging
    let log_file_path = match args.ensure_log_file_path().await {
        Ok(path) => path,
        Err(err) => {
            log::error!("Failed to create log file: {}", err);
            return;
        }
    };
    let stdout_env = env_logger::Builder::new()
        .filter_level(LevelFilter::from_str(&args.log_level).expect("Invalid log level"))
        .format_timestamp(None)
        .format_module_path(false)
        .build();
    let logger = AsyncProtobufLogger::init_with_env_logger(log_file_path.clone(), stdout_env);
    log::set_logger(logger).unwrap();
    log::set_max_level(log::LevelFilter::Debug);
    log::info!("Saving logs to {}", log_file_path.display());

    let (stop_tx, stop_rx) = broadcast::channel(1);
    let main_task = tokio::spawn(async move { Mode::run(args, stop_rx).await });

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");
    logger.flush();
    println!("Shutting down... Press ctrl-c again to force shutdown");
    // Allow the logger to flush before shutting down
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    let shutdown_fut = async move {
        stop_tx.send(()).expect("Failed to send stop signal");
        main_task.await.expect("Executor task failed");
    };
    tokio::select! {
        _ = shutdown_fut => {}
        _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
            eprintln!("Shutdown timed out");
            std::process::exit(1);
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Forced shutdown");
            std::process::exit(1);
        }
    };
}
