use anyhow::Context;
use anyhow::Result;
use clap::{Parser, ValueEnum};
use dies_core::workspace_utils;
use dies_serial_client::SerialClientConfig;
use dies_simulator::Simulation;
use dies_simulator::SimulationConfig;
use dies_ssl_client::VisionClientConfig;
use dies_world::WorldConfig;
use mock_vision::MockVision;
use std::net::SocketAddr;
use std::{path::PathBuf, str::FromStr};
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;

use dies_python_rt::{PyExecute, PyRuntimeConfig};
use dies_serial_client::list_serial_ports;

mod executor;
mod mock_vision;

use crate::executor::{run, ExecutorConfig};

#[derive(Debug, Clone, ValueEnum)]
enum VisionType {
    Tcp,
    Udp,
    Mock,
    Simulator,
}

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
struct Args {
    #[clap(long, default_value = "auto")]
    serial_port: String,

    #[clap(long, default_value = "true")]
    webui: bool,

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

    #[clap(long, default_value = "info")]
    log_level: String,

    #[clap(long, default_value = "auto")]
    log_file: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Set up log file
    let log_file_path = if args.log_file != "auto" {
        let path = PathBuf::from(args.log_file);
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

    let (vision_config, mut serial_config) = match args.vision {
        VisionType::Tcp => (
            VisionClientConfig::Tcp {
                host: args.vision_addr.ip().to_string(),
                port: args.vision_addr.port(),
            },
            None,
        ),
        VisionType::Udp => (
            VisionClientConfig::Udp {
                host: args.vision_addr.ip().to_string(),
                port: args.vision_addr.port(),
            },
            None,
        ),
        VisionType::Mock => (MockVision::spawn(), None),
        VisionType::Simulator => {
            tracing::info!("Using simulator");
            let (vision_config, serial) = Simulation::spawn(SimulationConfig::default());
            (vision_config, Some(serial))
        }
    };

    if serial_config.is_none() {
        let ports = list_serial_ports().context("Failed to list serial ports")?;
        let port = if args.serial_port != "false" {
            if args.serial_port != "auto" {
                if !ports.contains(&args.serial_port) {
                    eprintln!("Port {} not found", args.serial_port);
                    eprintln!(
                        "Available ports:\n{}",
                        ports
                            .iter()
                            .map(|p| format!("  - {}", p))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    std::process::exit(1);
                }
                Some(args.serial_port.clone())
            } else if ports.is_empty() {
                tracing::warn!("No serial ports found, disabling serial");
                None
            } else if ports.len() == 1 {
                tracing::info!("Connecting to serial port {}", ports[0]);
                Some(ports[0].clone())
            } else {
                println!("Available ports:");
                for (idx, port) in ports.iter().enumerate() {
                    println!("{}: {}", idx, port);
                }

                // Let user choose port
                loop {
                    println!("Enter port number:");
                    let mut input = String::new();
                    std::io::stdin().read_line(&mut input)?;
                    let port_idx = input
                        .trim()
                        .parse::<usize>()
                        .context("Failed to parse the input into a number (usize)")?;
                    if port_idx < ports.len() {
                        break Some(ports[port_idx].clone());
                    }
                    println!("Invalid port number");
                }
            }
        } else {
            tracing::warn!("Serial disabled");
            None
        };
        tracing::debug!("Serial port: {:?}", port);

        if let Some(port) = &port {
            serial_config = Some(SerialClientConfig::serial(port.clone()));
        }
    }

    let robot_ids = args
        .robot_ids
        .split(',')
        .filter_map(|s| {
            let mut parts = s.split(':');
            let id = parts.next()?.parse().ok()?;
            let team = parts.next()?.parse().ok()?;
            Some((id, team))
        })
        .collect();

    let workspace_root = workspace_utils::get_workspace_root();
    let config = ExecutorConfig {
        webui: args.webui,
        robot_ids,
        py_config: if !args.disable_python {
            Some(PyRuntimeConfig {
                install: true,
                workspace: workspace_root.clone(),
                python_build: 20240107,
                python_version: "3.11.7".into(),
                execute: PyExecute::Package {
                    path: workspace_root.join("strategy").join(&args.package),
                    name: args.package,
                },
            })
        } else {
            None
        },
        world_config: WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        },
        vision_config,
        serial_config,
    };

    run(config).await.expect("Failed to run executor");

    tracing::info!("Shutting down");

    Ok(())
}
