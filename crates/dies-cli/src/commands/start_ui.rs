use anyhow::{bail, Result};
use clap::Parser;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_ssl_client::{ConnectionConfig, SslClientConfig};
use dies_webui::{UiConfig, UiEnvironment};
use std::{net::SocketAddr, path::PathBuf, str::FromStr};

use dies_logger::AsyncProtobufLogger;
use log::{LevelFilter, Log};
use tokio::sync::broadcast;

use crate::cli::{BasestationProtocolVersion, ConnectionMode, SerialPort};

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
pub struct MainArgs {
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

    #[clap(long, default_value = "enp4s0")]
    pub interface: Option<String>,

    #[clap(long, default_value = "info")]
    pub log_level: String,

    #[clap(long, default_value = "logs")]
    pub log_directory: String,
}

impl MainArgs {
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
            .map(|port| {
                let mut config = BasestationClientConfig::new(port, self.protocol.into());
                config.set_robot_id_map_from_string(&self.robot_ids);
                config
            })
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

pub async fn start_ui(args: MainArgs) -> Result<()> {
    // Set up logging
    let log_dir_path = match args.ensure_log_dir_path().await {
        Ok(path) => path,
        Err(err) => {
            bail!("Failed to create log directory: {}", err);
        }
    };
    println!("Saving logs to {}", log_dir_path.display());
    let stdout_env = env_logger::Builder::new()
        .filter_level(LevelFilter::from_str(&args.log_level).expect("Invalid log level"))
        .format_timestamp(None)
        .format_module_path(false)
        .build();
    let logger = AsyncProtobufLogger::init_with_env_logger(log_dir_path.clone(), stdout_env);
    log::set_logger(logger).unwrap(); // Safe to unwrap because we know no logger has been set yet
    log::set_max_level(log::LevelFilter::Debug);

    let (stop_tx, stop_rx) = broadcast::channel(1);
    let main_task = tokio::spawn(async move {
        let conf = match args.into_ui().await {
            Ok(conf) => conf,
            Err(err) => {
                log::error!("Failed to parse UI configuration: {}", err);
                return;
            }
        };
        dies_webui::start(conf, stop_rx).await;
    });

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    logger.flush();
    println!("Shutting down (timeout 3 seconds)... Press ctrl-c again to force shutdown");
    // Allow the logger to flush before shutting down
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Fool-proof timeout for shutdown
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(3));
        eprintln!("Shutdown timed out");
        std::process::exit(1);
    });

    let shutdown_fut = async move {
        stop_tx.send(()).expect("Failed to send stop signal");
        main_task.await.expect("Executor task failed");
    };
    tokio::select! {
        _ = shutdown_fut => {}
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Forced shutdown");
            std::process::exit(1);
        }
    };

    Ok(())
}
