use anyhow::{bail, Result};
use dies_basestation_client::list_serial_ports;
use start_ui::MainArgs;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::cli::SerialPort;

pub mod convert_logs;
pub mod start_ui;
pub mod test_radio;

/// Selects a serial port based on the CLI arguments. This function may prompt the user
/// to choose a port if multiple ports are available and the `serial_port` argument is
/// set to "auto".
async fn select_serial_port(args: &MainArgs) -> Result<String> {
    let ports = list_serial_ports();
    let port = match &args.serial_port {
        SerialPort::Disabled => None,
        SerialPort::Auto => {
            if ports.is_empty() {
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
                    if let Some(input) = read_line().await {
                        if let Ok(port_idx) = input.trim().parse::<usize>() {
                            if port_idx < ports.len() {
                                break Some(ports[port_idx].clone());
                            }
                        } else {
                            println!("Invalid port number");
                        }
                    } else {
                        println!("Invalid port number");
                    }
                }
            }
        }
        SerialPort::Port(port) => {
            if !ports.contains(port) {
                println!(
                    "Available ports:\n{}",
                    ports
                        .iter()
                        .map(|p| format!("  - {}", p))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                bail!("Port {} not found", port);
            }
            Some(port.clone())
        }
    };

    port.ok_or(anyhow::anyhow!("Serial port not found"))
}

async fn read_line() -> Option<String> {
    let reader = BufReader::new(tokio::io::stdin());
    reader
        .lines()
        .next_line()
        .await
        .ok()
        .flatten()
        .map(|s| s.trim().to_owned())
}
