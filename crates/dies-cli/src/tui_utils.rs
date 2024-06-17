use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use dies_serial_client::{list_serial_ports, SerialClientConfig};
use dies_ssl_client::VisionClientConfig;
use dies_webui::{UiConfig, UiEnvironment};
use std::{net::SocketAddr, path::PathBuf};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::modes::Mode;

#[derive(Debug, Clone, ValueEnum, Default)]
pub enum VisionType {
    #[default]
    None,
    Tcp,
    Udp,
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

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
pub struct CliArgs {
    #[clap(long, short, default_value = "ui")]
    pub mode: Mode,

    #[clap(long, default_value = "disabled", default_missing_value = "auto")]
    pub serial_port: SerialPort,

    #[clap(long, default_value = "true")]
    pub webui: bool,

    #[clap(long, default_value = "5555")]
    pub webui_port: u16,

    #[clap(long, default_value = "false")]
    pub webui_devserver: bool,

    #[clap(long, default_value = "false")]
    pub disable_python: bool,

    #[clap(long, default_value = "")]
    pub robot_ids: String,

    #[clap(long, default_value = "dies-test-strat")]
    pub package: String,

    #[clap(long, default_value = "none")]
    pub vision: VisionType,

    #[clap(long, default_value = "224.5.23.2:10006")]
    pub vision_addr: SocketAddr,

    #[clap(long, default_value = "info")]
    pub log_level: String,

    #[clap(long, default_value = "auto")]
    pub log_file: String,
}

impl CliArgs {
    /// Converts the CLI arguments into a `UiConfig` object that can be used to start the web UI.
    pub async fn into_ui(self) -> UiConfig {
        let environment = match (self.serial_config().await, self.vision_config()) {
            (Some(bs_config), Some(ssl_config)) => UiEnvironment::WithLive {
                bs_config,
                ssl_config,
            },
            _ => UiEnvironment::SimulationOnly,
        };

        UiConfig {
            environment,
            port: self.webui_port,
        }
    }

    /// Configures the serial client based on the CLI arguments. This function may prompt the user
    /// to choose a port if multiple ports are available and the `serial_port` argument is set to "auto".
    ///
    /// If there is an issue selecting a serial port, an error message will be logged and `None` will be returned.
    pub async fn serial_config(&self) -> Option<SerialClientConfig> {
        select_serial_port(self)
            .await
            .map_err(|err| log::warn!("Failed to setup serial: {}", err))
            .ok()
            .map(|port| {
                let mut config = SerialClientConfig::new(port);
                config.set_robot_id_map_from_string(&self.robot_ids);
                config
            })
    }

    /// Configures the vision client based on the CLI arguments.
    pub fn vision_config(&self) -> Option<VisionClientConfig> {
        match self.vision {
            VisionType::None => None,
            VisionType::Tcp => Some(VisionClientConfig::Tcp {
                host: self.vision_addr.ip().to_string(),
                port: self.vision_addr.port(),
            }),
            VisionType::Udp => Some(VisionClientConfig::Udp {
                host: self.vision_addr.ip().to_string(),
                port: self.vision_addr.port(),
            }),
        }
    }

    /// Returns the path to the log file. If the `log_file` argument is set to "auto", a
    /// timestamped log file will be created in the user's data directory.
    ///
    /// This function will make sure the log file does not already exist and will create the
    /// necessary directories if needed.
    pub async fn ensure_log_file_path(&self) -> Result<PathBuf> {
        if self.log_file != "auto" {
            let path = PathBuf::from(self.log_file.clone());
            if path.exists() {
                bail!("Log file already exists: {}", path.display());
            }
            Ok(path)
        } else {
            let time = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
            let filemame = format!("dies-{time}.log");
            let path = dirs::data_local_dir()
                .map(|p| p.join("dies").join(&filemame))
                .unwrap_or_else(|| PathBuf::from(&filemame));
            let dir = path.parent().unwrap();
            tokio::fs::create_dir_all(dir).await?;
            Ok(path)
        }
    }
}

/// Selects a serial port based on the CLI arguments. This function may prompt the user
/// to choose a port if multiple ports are available and the `serial_port` argument is
/// set to "auto".
pub async fn select_serial_port(args: &CliArgs) -> Result<String> {
    let ports = list_serial_ports().context("Failed to list serial ports")?;
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
