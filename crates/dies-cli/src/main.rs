use anyhow::Result;
use clap::Parser;
use dies_core::workspace_utils;
use dies_ssl_client::SslVisionClientConfig;
use dies_world::WorldConfig;

use dies_python_rt::{PyExecute, PyRuntimeConfig};
use dies_serial_client::list_serial_ports;
use tokio_util::sync::CancellationToken;

mod executor;
mod run_on_server;

use crate::executor::{run, ExecutorConfig};

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
struct Args {
    #[clap(long, default_value = "false")]
    run_on_server: bool,

    #[clap(long, default_value = "false")]
    sync: bool,

    #[clap(long, default_value = None)]
    serial_port: Option<String>,

    #[clap(long, default_value = "false")]
    webui: bool,

    #[clap(long, default_value = "14:3,5:2")]
    robot_ids: String,

    #[clap(long, default_value = "dies-test-strat")]
    package: String,

    #[clap(long, default_value = "localhost")]
    vision_host: String,

    #[clap(long, default_value = "6078")]
    vision_port: u16,

    #[clap(long, default_value = "tcp")]
    vision_socket_type: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    if args.run_on_server {
        println!("Executing on server");
        run_on_server::run_on_server();
        return Ok(());
    }

    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .init();

    let ports = list_serial_ports()?;

    let port = if !ports.is_empty() {
        if let Some(port) = &args.serial_port {
            if !ports.contains(port) {
                return Err(anyhow::anyhow!("Port {} not found", port));
            }
            Some(port.clone())
        } else if ports.len() == 1 {
            log::info!("Connecting to serial port {}", ports[0]);
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
                let port_idx = input.trim().parse::<usize>()?;
                if port_idx < ports.len() {
                    break Some(ports[port_idx].clone());
                }
                println!("Invalid port number");
            }
        }
    } else {
        log::warn!("No serial ports available, not connecting to basestation");
        None
    };

    let vision_config = SslVisionClientConfig {
        socket_type: if args.vision_socket_type == "udp" {
            dies_ssl_client::SocketType::Udp
        } else {
            dies_ssl_client::SocketType::Tcp
        },
        host: args.vision_host,
        port: args.vision_port,
    };

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
        webui: false,
        robot_ids,
        py_config: PyRuntimeConfig {
            install: true,
            workspace: workspace_root.clone(),
            python_build: 20240107,
            python_version: "3.11.7".into(),
            execute: PyExecute::Package {
                path: workspace_root.join("strategy").join(&args.package),
                name: args.package,
            },
        },
        world_config: WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        },
        vision_config: Some(vision_config),
        serial_config: match port {
            Some(port) => Some(dies_serial_client::SerialClientConfig {
                port_name: port.clone(),
                ..Default::default()
            }),
            None => None,
        },
    };

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    run(config, cancel_clone.clone())
        .await
        .expect("Failed to run executor");

    log::info!("Shutting down");

    Ok(())
}
