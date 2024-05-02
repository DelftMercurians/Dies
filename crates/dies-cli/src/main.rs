use anyhow::Context;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use dies_core::workspace_utils;
use std::net::SocketAddr;
use std::{path::PathBuf, str::FromStr};
use tokio::sync::broadcast;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;

mod mock_vision;
mod modes;

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum VisionType {
    Tcp,
    Udp,
    Mock,
}

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
pub(crate) struct Args {
    #[clap(long, short)]
    mode: modes::Mode,

    #[clap(long, default_value = "auto")]
    serial_port: String,

    #[clap(long, default_value = "true")]
    webui: bool,

    #[clap(long, default_value = "false")]
    webui_devserver: bool,

    #[clap(long, default_value = "false")]
    disable_python: bool,

    #[clap(long, default_value = "14:3,5:2")]
    robot_ids: String,

    #[clap(long, default_value = "dies-test-strat")]
    package: String,

    #[clap(long, default_value = "udp")]
    vision: VisionType,

    #[clap(long, default_value = "224.5.23.2:10006")]
    vision_addr: SocketAddr,

    #[clap(long, default_value = "debug")]
    log_level: String,

    #[clap(long, default_value = "auto")]
    log_file: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Set up log file
    let log_file_path = if args.log_file != "auto" {
        let path = PathBuf::from(args.log_file.clone());
        if path.exists() {
            eprintln!("Log file already exists: {}", path.display());
            std::process::exit(1);
        }
        path
    } else {
        let time = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
        let filemame = format!("dies-{time}.log");
        let path = dirs::data_local_dir()
            .map(|p| p.join("dies").join(&filemame))
            .unwrap_or_else(|| PathBuf::from(&filemame));
        let dir = path.parent().unwrap();
        tokio::fs::create_dir_all(dir).await.expect(&format!(
            "Failed to create log directory: {}",
            dir.display()
        ));
        path
    };

    // Create log file appender
    let appender = tracing_appender::rolling::never(
        log_file_path.parent().unwrap(),
        log_file_path.file_name().unwrap(),
    );
    let (non_blocking_appender, _guard) = tracing_appender::non_blocking(appender);

    // Set up tracing
    let log_level = match tracing::Level::from_str(&args.log_level) {
        Ok(level) => level,
        Err(_) => {
            eprintln!("Invalid log level: {}", args.log_level);
            std::process::exit(1);
        }
    };
    let stdout_layer = fmt::Subscriber::builder()
        .with_max_level(log_level)
        .without_time()
        .finish();
    let logfile_layer = fmt::Layer::default()
        .json()
        .with_ansi(false)
        .with_writer(non_blocking_appender);
    tracing::subscriber::set_global_default(stdout_layer.with(logfile_layer))
        .expect("Unable to set global tracing subscriber");

    tracing::info!("Saving logs to {}", log_file_path.display());

    let devserver = if args.webui_devserver {
        let workspace_root = workspace_utils::get_workspace_root();
        let child = tokio::process::Command::new("npm")
            .args(&[
                "run",
                "--silent",
                "dev",
                "--",
                "--clearScreen",
                "false",
                "--logLevel",
                "error",
                "--port",
                "5173",
            ])
            .current_dir(workspace_root.join("webui"))
            .spawn()
            .context("Failed to start webui dev server")?;
        println!("Started webui dev server at {}", "http://localhost:5173");
        Some(child)
    } else {
        None
    };

    let (stop_tx, stop_rx) = broadcast::channel(1);
    let main_task = tokio::spawn(async move {
        let result = match args.mode {
            modes::Mode::Irl => modes::irl::run(args, stop_rx).await,
            modes::Mode::SimTest => modes::sim_test::run(stop_rx).await,
        };
        if let Err(err) = result {
            tracing::error!("Mode failed: {}", err);
        }
    });

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    tracing::info!("Shutting down");
    stop_tx.send(()).expect("Failed to send stop signal");
    main_task.await.expect("Executor task failed");

    if let Some(mut child) = devserver {
        child.kill().await.expect("Failed to kill dev server");
    }

    Ok(())
}
