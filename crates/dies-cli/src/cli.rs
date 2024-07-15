use anyhow::{bail, Result};
use dies_basestation_client::{list_serial_ports, BasestationClientConfig, BasestationHandle};
use dies_ssl_client::VisionClientConfig;
use dies_webui::{UiConfig, UiEnvironment};
use std::net::SocketAddr;
use std::{path::PathBuf, process::ExitCode};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::commands::start_ui::MainArgs;
use crate::commands::{convert_logs::convert_log, start_ui::start_ui};
use clap::{Parser, Subcommand, ValueEnum};

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

#[derive(Debug, Clone, Subcommand)]
enum Command {
    #[clap(name = "convert")]
    Convert {
        #[clap(short, long)]
        input: PathBuf,
        #[clap(short, long)]
        output: PathBuf,
    },
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

