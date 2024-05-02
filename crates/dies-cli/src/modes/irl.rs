use crate::mock_vision::MockVision;
use crate::VisionType;
use anyhow::Context;
use anyhow::Result;
use dies_executor::Executor;
use dies_serial_client::SerialClient;
use dies_serial_client::SerialClientConfig;
use dies_ssl_client::VisionClient;
use dies_ssl_client::VisionClientConfig;

use dies_serial_client::list_serial_ports;
use tokio::sync::broadcast;

pub async fn run(args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let vision_config = match args.vision {
        VisionType::Tcp => VisionClientConfig::Tcp {
            host: args.vision_addr.ip().to_string(),
            port: args.vision_addr.port(),
        },
        VisionType::Udp => VisionClientConfig::Udp {
            host: args.vision_addr.ip().to_string(),
            port: args.vision_addr.port(),
        },
        VisionType::Mock => MockVision::spawn(),
    };

    let serial_client = {
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
            Some(SerialClient::new(SerialClientConfig::serial(port.clone()))?)
        } else {
            None
        }
    };

    let mut builder = Executor::builder();
    builder.with_bs_client(serial_client.unwrap());
    builder.with_ssl_client(VisionClient::new(vision_config).await?);

    let executor = builder.build()?;
    executor.run_real_time(stop_rx).await
}
