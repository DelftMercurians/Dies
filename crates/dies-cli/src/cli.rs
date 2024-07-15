use std::net::SocketAddr;
use std::{path::PathBuf, process::ExitCode};

use crate::commands::start_ui::MainArgs;
use crate::commands::test_radio::test_radio;
use crate::commands::test_vision::test_vision;
use crate::commands::{convert_logs::convert_log, start_ui::start_ui};
use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dies_basestation_client::list_serial_ports;
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Debug, Clone, Subcommand)]
enum Command {
    #[clap(name = "convert")]
    Convert {
        #[clap(short, long)]
        input: PathBuf,
        #[clap(short, long)]
        output: PathBuf,
    },

    /// Convert the last log file in the logs directory to JSON, writing the output to log.json.
    #[clap(name = "convert-last")]
    ConvertLast,

    #[clap(name = "test-radio")]
    TestRadio {
        #[clap(short, long, default_value = "auto")]
        port: SerialPort,

        #[clap(long, default_value = "3.0")]
        duration: f64,
        #[clap(long)]
        w: Option<f64>,
        #[clap(long)]
        sx: Option<f64>,
        #[clap(long)]
        sy: Option<f64>,

        /// The IDs of the robots to test.
        ids: Vec<u32>,
    },

    #[clap(name = "test-vision")]
    TestVision {
        #[clap(short, long, default_value = "udp")]
        vision: VisionType,
        #[clap(long, default_value = "224.5.23.2:10006")]
        vision_addr: SocketAddr,
    },
}

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
pub struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,
}

impl Cli {
    pub async fn start() -> ExitCode {
        let cli = Cli::parse();
        match cli.command {
            Some(Command::Convert { input, output }) => match convert_log(&input, &output) {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error converting logs: {}", err);
                    ExitCode::FAILURE
                }
            },
            Some(Command::ConvertLast) => {
                match crate::commands::convert_last_log::convert_last_log() {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error converting logs: {}", err);
                        ExitCode::FAILURE
                    }
                }
            }
            Some(Command::TestRadio {
                port,
                duration,
                ids: id,
                w,
                sx,
                sy,
            }) => match test_radio(port, id, duration, w, sx, sy).await {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error testing radio: {}", err);
                    ExitCode::FAILURE
                }
            },
            Some(Command::TestVision {
                vision,
                vision_addr,
            }) => match test_vision(vision, vision_addr).await {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error testing vision: {}", err);
                    ExitCode::FAILURE
                }
            },
            None => {
                let args = MainArgs::parse();
                match start_ui(args).await {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error in UI: {}", err);
                        ExitCode::FAILURE
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum VisionType {
    #[default]
    None,
    Tcp,
    Udp,
}

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum BasestationProtocolVersion {
    #[default]
    V1,
    V0,
}

impl From<BasestationProtocolVersion> for dies_basestation_client::BaseStationProtocol {
    fn from(val: BasestationProtocolVersion) -> Self {
        match val {
            BasestationProtocolVersion::V0 => dies_basestation_client::BaseStationProtocol::V0,
            BasestationProtocolVersion::V1 => dies_basestation_client::BaseStationProtocol::V1,
        }
    }
}

/// The serial port to connect to.
#[derive(Debug, Clone, Default)]
pub enum SerialPort {
    #[default]
    Disabled,
    Auto,
    Port(String),
}

impl SerialPort {
    const VARIANTS: &'static [Self] = &[
        SerialPort::Disabled,
        SerialPort::Auto,
        SerialPort::Port(String::new()),
    ];

    pub async fn select(&self) -> Result<String> {
        select_serial_port(self).await
    }
}

impl ValueEnum for SerialPort {
    fn value_variants<'a>() -> &'a [Self] {
        Self::VARIANTS
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            SerialPort::Disabled => Some(clap::builder::PossibleValue::new("disabled")),
            SerialPort::Auto => Some(clap::builder::PossibleValue::new("auto")),
            SerialPort::Port(port) => Some(clap::builder::PossibleValue::new(port)),
        }
    }

    fn from_str(input: &str, ignore_case: bool) -> std::result::Result<Self, String> {
        if ignore_case {
            match input.to_lowercase().as_str() {
                "disabled" => Ok(SerialPort::Disabled),
                "auto" => Ok(SerialPort::Auto),
                _ => Ok(SerialPort::Port(input.to_owned())),
            }
        } else {
            match input {
                "disabled" => Ok(SerialPort::Disabled),
                "auto" => Ok(SerialPort::Auto),
                _ => Ok(SerialPort::Port(input.to_owned())),
            }
        }
    }
}

/// Selects a serial port based on the CLI arguments. This function may prompt the user
/// to choose a port if multiple ports are available and the `serial_port` argument is
/// set to "auto".
async fn select_serial_port(serial_port: &SerialPort) -> Result<String> {
    let ports = list_serial_ports();
    let port = match serial_port {
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
