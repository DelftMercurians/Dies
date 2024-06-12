use anyhow::{Context, Result};
use dies_serial_client::{list_serial_ports, SerialClient, SerialClientConfig};
use dies_ssl_client::{VisionClient, VisionClientConfig};

use crate::{Args, VisionType};

pub async fn setup_vision_and_serial(args: &Args) -> Result<(VisionClient, Option<SerialClient>)> {
    let vision_config = match args.vision {
        VisionType::Tcp => VisionClientConfig::Tcp {
            host: args.vision_addr.ip().to_string(),
            port: args.vision_addr.port(),
        },
        VisionType::Udp => VisionClientConfig::Udp {
            host: args.vision_addr.ip().to_string(),
            port: args.vision_addr.port(),
        },
    };
    let vision = VisionClient::new(vision_config).await?;

    let serial = {
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
                log::warn!("No serial ports found, disabling serial");
                None
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
            log::warn!("Serial disabled");
            None
        };
        log::debug!("Serial port: {:?}", port);

        if let Some(port) = &port {
            let mut config = SerialClientConfig::new(port.clone());
            config.set_robot_id_map_from_string(&args.robot_ids);
            Some(SerialClient::new(config)?)
        } else {
            None
        }
    };

    Ok((vision, serial))
}
