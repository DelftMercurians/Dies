use std::{net::SocketAddr, path::PathBuf, process::ExitCode};

use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use dies_basestation_client::{list_serial_ports, BasestationClientConfig, BasestationHandle};
use dies_ssl_client::{ConnectionConfig, SslClientConfig};
use dies_webui::{UiConfig, UiEnvironment};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::commands::{
    convert_logs::convert_log, start_ui::start_ui, test_radio::test_radio, test_vision::test_vision,
};

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
        #[clap(long, default_value = "3.0")]
        duration: f64,
        #[clap(long, allow_hyphen_values = true)]
        w: Option<f64>,
        #[clap(long, allow_hyphen_values = true)]
        sx: Option<f64>,
        #[clap(long, allow_hyphen_values = true)]
        sy: Option<f64>,

        #[clap(long, default_value = "10000")]
        max_yaw_rate: f64,
        #[clap(long, default_value = "0", allow_hyphen_values = true)]
        preferred_rotation_direction: f64,

        #[clap(long, default_value = "false", action)]
        kick: bool,

        /// The IDs of the robots to test.
        ids: Vec<u32>,
    },

    #[clap(name = "test-vision")]
    TestVision,
}

#[derive(Debug, Clone, Serialize, Deserialize, ValueEnum)]
pub enum ControlledTeam {
    Blue,
    Yellow,
    Both,
}

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
pub struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,

    #[clap(long, short = 'f', default_value = "dies-settings.json")]
    pub settings_file: PathBuf,

    #[clap(long, default_value = "auto", default_missing_value = "auto")]
    pub serial_port: SerialPort,

    #[clap(long, default_value = "v1")]
    pub protocol: BasestationProtocolVersion,

    #[clap(long, default_value = "5555")]
    pub webui_port: u16,

    #[clap(long, default_value = "false")]
    pub disable_python: bool,

    #[clap(long, default_value = "")]
    pub robot_ids: String,

    #[clap(long, default_value = "udp")]
    pub mode: ConnectionMode,

    #[clap(long, default_value = "224.5.23.2:10006")]
    pub vision_addr: SocketAddr,

    #[clap(long, default_value = "224.5.23.1:10003")]
    pub gc_addr: SocketAddr,

    // current interface enp3s0f1 |  previous interface enxf8e43ba77d03
    #[clap(long, default_value = "enp3s0f1")]
    pub interface: Option<String>,

    #[clap(long, default_value = "info")]
    pub log_level: String,

    #[clap(long, default_value = "logs")]
    pub log_directory: String,

    #[clap(long, default_value = "simulation")]
    pub ui_mode: String,

    #[clap(long, default_value = "false", action)]
    pub auto_start: bool,

    #[clap(long, default_value = "blue")]
    pub controlled_teams: ControlledTeam,

    #[clap(long, default_value = "v0")]
    pub strategy: String,
}

impl Cli {
    pub async fn start(self) -> ExitCode {
        match self.command {
            None => match start_ui(self).await {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error in UI: {}", err);
                    ExitCode::FAILURE
                }
            },
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
                duration,
                ids: id,
                w,
                sx,
                sy,
                max_yaw_rate,
                preferred_rotation_direction,
                kick,
            }) => match test_radio(
                self.serial_port,
                id,
                duration,
                w,
                sx,
                sy,
                max_yaw_rate,
                preferred_rotation_direction,
                kick,
            )
            .await
            {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error testing radio: {}", err);
                    ExitCode::FAILURE
                }
            },
            Some(Command::TestVision) => {
                match test_vision(self.mode, self.vision_addr, self.gc_addr, self.interface).await {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error testing vision: {}", err);
                        // eprintln!("Mode: {}", self.mode);
                        eprintln!("vision_addr: {}", self.vision_addr);
                        eprintln!("gc_addr: {}", self.gc_addr);
                        // eprintln!("interface: {}", self.interface);

                        ExitCode::FAILURE
                    }
                }
            }
        }
    }

    /// Converts the CLI arguments into a `UiConfig` object that can be used to start the web UI.
    pub async fn into_ui(self) -> Result<UiConfig> {
        let environment = match (self.serial_config().await, self.ssl_config()) {
            (Some(bs_config), Some(ssl_config)) => UiEnvironment::WithLive {
                bs_handle: BasestationHandle::spawn(bs_config)?,
                ssl_config,
            },
            _ => UiEnvironment::SimulationOnly,
        };

        Ok(UiConfig {
            settings_file: self.settings_file,
            environment,
            port: self.webui_port,
            start_mode: match self.ui_mode.as_str() {
                "simulation" => dies_webui::UiMode::Simulation,
                "live" => dies_webui::UiMode::Live,
                _ => {
                    bail!("Invalid UI mode: {}", self.ui_mode);
                }
            },
            auto_start: self.auto_start,
            controlled_teams: match self.controlled_teams {
                ControlledTeam::Blue => dies_webui::ControlledTeam::Blue,
                ControlledTeam::Yellow => dies_webui::ControlledTeam::Yellow,
                ControlledTeam::Both => dies_webui::ControlledTeam::Both,
            },
        })
    }

    /// Configures the serial client based on the CLI arguments. This function may prompt the user
    /// to choose a port if multiple ports are available and the `serial_port` argument is set to "auto".
    ///
    /// If there is an issue selecting a serial port, an error message will be logged and `None` will be returned.
    pub async fn serial_config(&self) -> Option<BasestationClientConfig> {
        self.serial_port
            .select()
            .await
            .map_err(|err| log::warn!("Failed to setup serial: {}", err))
            .ok()
            .map(|port| BasestationClientConfig::new(port, self.protocol.into()))
    }

    /// Configures the vision client based on the CLI arguments.
    pub fn ssl_config(&self) -> Option<SslClientConfig> {
        let vision = match self.mode {
            ConnectionMode::None => None,
            ConnectionMode::Tcp => Some(ConnectionConfig::Tcp {
                host: self.vision_addr.ip().to_string(),
                port: self.vision_addr.port(),
            }),
            ConnectionMode::Udp => Some(ConnectionConfig::Udp {
                host: self.vision_addr.ip().to_string(),
                port: self.vision_addr.port(),
                interface: self.interface.clone(),
            }),
        };
        let gc = match self.mode {
            ConnectionMode::None => None,
            ConnectionMode::Tcp => Some(ConnectionConfig::Tcp {
                host: self.gc_addr.ip().to_string(),
                port: self.gc_addr.port(),
            }),
            ConnectionMode::Udp => Some(ConnectionConfig::Udp {
                host: self.gc_addr.ip().to_string(),
                port: self.gc_addr.port(),
                interface: self.interface.clone(),
            }),
        };

        match (vision, gc) {
            (Some(vision), Some(gc)) => Some(SslClientConfig { vision, gc }),
            _ => {
                log::warn!("Invalid SSL configuration");
                None
            }
        }
    }

    /// Returns the path to the log directory, making sure it exists. Defaults to "logs" in the current directory.
    pub async fn ensure_log_dir_path(&self) -> Result<PathBuf> {
        let path = PathBuf::from(&self.log_directory);
        tokio::fs::create_dir_all(&path).await?;
        Ok(path)
    }
}

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum ConnectionMode {
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
