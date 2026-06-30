use std::{collections::HashMap, net::SocketAddr, path::PathBuf, process::ExitCode, str::FromStr};

use anyhow::{bail, Result};
use clap::{ArgGroup, Parser, Subcommand, ValueEnum};
use dies_basestation_client::{list_serial_ports, BasestationClientConfig, BasestationHandle};
use dies_core::{ParamValue, PlayerId, StrategyParams, TeamColor};
use dies_ssl_client::{ConnectionConfig, SslClientConfig};
use dies_webui::{UiConfig, UiEnvironment};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::commands::{
    self_play::self_play, start_ui::start_ui, test_radio::test_radio, test_vision::test_vision,
};

/// How the strategy binary is managed for the duration of the run. Derived from
/// the mutually-exclusive `--launch` / `--build` / `--watch` flags.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum StrategyMode {
    /// Launch the existing binary as-is; never build.
    Launch,
    /// Build the strategy once before launching; abort on build failure.
    #[default]
    Build,
    /// Build once, then watch the sources and hot-swap the running strategy on
    /// every successful rebuild (dev mode).
    Watch,
}

#[derive(Debug, Clone, Subcommand)]
enum Command {
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

    /// Run one deterministic, faster-than-realtime headless A-vs-B self-play
    /// match in simulation (no webui). Emits a MatchResult as JSON.
    #[clap(name = "self-play")]
    SelfPlay {
        /// Strategy binary for the blue team.
        #[clap(long)]
        blue_strategy: String,

        /// Strategy binary for the yellow team.
        #[clap(long)]
        yellow_strategy: String,

        /// RNG seed driving initial pose jitter (same seed → identical match).
        #[clap(long, default_value = "0")]
        seed: u64,

        /// Match length in simulated seconds.
        #[clap(long, default_value = "120")]
        duration: f64,

        /// Stop early once the combined score reaches this many goals.
        #[clap(long)]
        max_goals: Option<u32>,

        /// Write the MatchResult JSON to this file instead of stdout.
        #[clap(long)]
        output: Option<PathBuf>,

        /// Record a full binary match log under this directory (for the
        /// analytics harness). Omit to skip logging.
        #[clap(long)]
        log_dir: Option<PathBuf>,

        /// Seed the field from a saved snapshot before kickoff. Accepts a
        /// snapshot name (resolved to `.dies-snapshots/<name>.json`) or a path
        /// to a snapshot JSON file. Robot poses + ball are teleported into
        /// place and the game state is forced to the snapshot's state (or Run
        /// if it has none), skipping the normal kickoff sequence.
        #[clap(long)]
        snapshot: Option<String>,

        /// Build the strategies in release profile (faster matches) instead of
        /// debug. Implies launching from `target/release` unless
        /// `--strategies-dir` overrides it.
        #[clap(long, default_value = "false")]
        release_strategies: bool,

        /// Directory the executor launches strategy binaries from. Defaults to
        /// `target/release` with `--release-strategies`, else `target/debug`.
        /// Set to a directory of prebuilt binaries to run without rebuilding.
        #[clap(long)]
        strategies_dir: Option<PathBuf>,

        /// Number of robots to field for the blue team (1..=6, default 6).
        #[clap(long, default_value = "6")]
        blue_robots: usize,

        /// Number of robots to field for the yellow team (1..=6, default 6).
        #[clap(long, default_value = "6")]
        yellow_robots: usize,

        /// Show the blue team a yellow card at this sim time (seconds). Repeat
        /// for multiple cards; each lowers blue's bot allowance for 120 s.
        #[clap(long)]
        blue_card_at: Vec<f64>,

        /// Show the yellow team a yellow card at this sim time (seconds). Repeat
        /// for multiple cards; each lowers yellow's bot allowance for 120 s.
        #[clap(long)]
        yellow_card_at: Vec<f64>,
    },

    /// Supervised match mode: run a pre-match checklist, then launch dies and
    /// auto-restart it on any crash. Starts concerto with `warmup=true`.
    #[clap(name = "match")]
    Match {
        /// Pre-match checklist markdown file (one item per line).
        #[clap(long, default_value = "match-checklist.md")]
        checklist: PathBuf,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, ValueEnum)]
pub enum ControlledTeam {
    Blue,
    Yellow,
    Both,
}

#[derive(Debug, Parser)]
#[command(name = "dies-cli")]
#[command(group(ArgGroup::new("strategy_mode").args(["launch", "build", "watch"]).multiple(false)))]
pub struct Cli {
    #[clap(subcommand)]
    command: Option<Command>,

    #[clap(long, short = 'f', default_value = "dies-settings.json")]
    pub settings_file: PathBuf,

    #[clap(long, default_value = "auto", default_missing_value = "auto")]
    pub serial_port: SerialPort,

    #[clap(long, default_value = "5555")]
    pub webui_port: u16,

    #[clap(long, default_value = "false")]
    pub disable_python: bool,

    #[clap(long, default_value = "")]
    pub robot_ids: String,

    #[clap(long, default_value = "udp")]
    pub connection_mode: ConnectionMode,

    #[clap(long, default_value = "224.5.23.2:10006")]
    pub vision_addr: SocketAddr,

    #[clap(long, default_value = "224.5.23.1:10003")]
    pub gc_addr: SocketAddr,

    #[clap(long, default_value = "0")]
    pub vision_delay_ms: u32,

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

    /// Strategy binary name (e.g. "concerto"). Use "none" for no strategy.
    /// Applies to every controlled team; for an asymmetric matchup (e.g. a
    /// benchmark) use `--blue-strategy` / `--yellow-strategy` instead.
    #[clap(long, default_value = "none")]
    pub strategy: String,

    /// Strategy binary for the blue team only. Overrides `--strategy` for blue
    /// and activates the blue team. Pair with `--yellow-strategy` to pit two
    /// strategies against each other in sim.
    #[clap(long)]
    pub blue_strategy: Option<String>,

    /// Strategy binary for the yellow team only. Overrides `--strategy` for
    /// yellow and activates the yellow team.
    #[clap(long)]
    pub yellow_strategy: Option<String>,

    /// Seed an initial strategy parameter, `KEY=VALUE` (repeatable). Applied to
    /// every controlled team at executor build. The value is parsed as a bool
    /// (`true`/`false`), else an integer, else a float, else a string. Used by
    /// match mode to start concerto with `warmup=true`.
    #[clap(long = "strategy-param", value_name = "KEY=VALUE")]
    pub strategy_param: Vec<String>,

    /// Launch the existing strategy binary as-is; never build.
    #[clap(long, action)]
    pub launch: bool,

    /// Build the strategy once before launching, aborting on failure (default).
    #[clap(long, action)]
    pub build: bool,

    /// Build once, then watch the sources and hot-swap the running strategy on
    /// every successful rebuild (dev mode).
    #[clap(long, action)]
    pub watch: bool,

    #[clap(long, default_value = "false", action)]
    pub calibration_mode: bool,
}

impl Cli {
    /// Resolve the strategy management mode from the mutually-exclusive
    /// `--launch` / `--build` / `--watch` flags (defaults to `Build`). Exclusivity
    /// is enforced by clap via the `strategy_mode` arg group.
    pub fn strategy_mode(&self) -> StrategyMode {
        if self.watch {
            StrategyMode::Watch
        } else if self.launch {
            StrategyMode::Launch
        } else {
            StrategyMode::Build
        }
    }

    pub async fn start(self) -> ExitCode {
        match self.command {
            None => match start_ui(self).await {
                Ok(_) => ExitCode::SUCCESS,
                Err(err) => {
                    eprintln!("Error in UI: {}", err);
                    ExitCode::FAILURE
                }
            },
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
                match test_vision(
                    self.connection_mode,
                    self.vision_addr,
                    self.gc_addr,
                    self.interface,
                )
                .await
                {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error testing vision: {}", err);
                        eprintln!("vision_addr: {}", self.vision_addr);
                        eprintln!("gc_addr: {}", self.gc_addr);

                        ExitCode::FAILURE
                    }
                }
            }
            Some(Command::SelfPlay {
                ref blue_strategy,
                ref yellow_strategy,
                seed,
                duration,
                max_goals,
                ref output,
                ref log_dir,
                ref snapshot,
                release_strategies,
                ref strategies_dir,
                blue_robots,
                yellow_robots,
                ref blue_card_at,
                ref yellow_card_at,
            }) => {
                let build = self.strategy_mode() != StrategyMode::Launch;
                match self_play(
                    blue_strategy.clone(),
                    yellow_strategy.clone(),
                    seed,
                    duration,
                    max_goals,
                    output.clone(),
                    log_dir.clone(),
                    snapshot.clone(),
                    build,
                    release_strategies,
                    strategies_dir.clone(),
                    blue_robots,
                    yellow_robots,
                    blue_card_at.clone(),
                    yellow_card_at.clone(),
                )
                .await
                {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error running self-play match: {}", err);
                        ExitCode::FAILURE
                    }
                }
            }
            Some(Command::Match { ref checklist }) => {
                let checklist = checklist.clone();
                match crate::commands::match_mode::run(self, checklist).await {
                    Ok(_) => ExitCode::SUCCESS,
                    Err(err) => {
                        eprintln!("Error in match mode: {}", err);
                        ExitCode::FAILURE
                    }
                }
            }
        }
    }

    /// Converts the CLI arguments into a `UiConfig` object that can be used to start the web UI.
    pub async fn into_ui(self) -> Result<UiConfig> {
        let calibration_mode = self.calibration_mode;
        let hot_reload = self.strategy_mode() == StrategyMode::Watch;
        // A live basestation only needs the serial config. Vision config is
        // optional — without it the executor can't run a Live match (unless
        // allow_no_vision), but the test bench still works.
        let environment = match self.serial_config().await {
            Some(bs_config) => UiEnvironment::WithLive {
                bs_handle: BasestationHandle::spawn(bs_config)?,
                ssl_config: self.ssl_config(),
            },
            None => UiEnvironment::SimulationOnly,
        };

        let strategy = if self.strategy == "none" {
            None
        } else {
            Some(self.strategy.clone())
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
            calibration_mode,
            strategy,
            blue_strategy: self.blue_strategy.clone(),
            yellow_strategy: self.yellow_strategy.clone(),
            hot_reload,
            vision_delay_ms: self.vision_delay_ms,
            log_directory: PathBuf::from(self.log_directory),
            initial_strategy_params: parse_strategy_params(&self.strategy_param)?,
        })
    }

    fn parse_robot_ids(&self) -> HashMap<(TeamColor, PlayerId), u32> {
        // eg. "blue:0-0,blue:1-1,blue:2-2,yellow:0-3,yellow:1-4,yellow:2-5"
        if self.robot_ids.is_empty() {
            return HashMap::new();
        }

        let mut map = HashMap::new();
        for id in self.robot_ids.split(',') {
            let (color, rest) = id.split_once(':').unwrap();
            let (player_id, robot_id) = rest.split_once('-').unwrap();
            let color = match color.to_lowercase().as_str() {
                "blue" => TeamColor::Blue,
                "yellow" => TeamColor::Yellow,
                _ => panic!("Invalid color: {}", color),
            };
            let player_id: u32 = player_id.parse().unwrap();
            let player_id = PlayerId::new(player_id);
            let robot_id: u32 = robot_id.parse().unwrap();
            map.insert((color, player_id), robot_id);
        }
        println!("parsed robot ids: {:?}", map);
        map
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
            .map(|port| BasestationClientConfig::new(port, self.parse_robot_ids()))
    }

    /// Configures the vision client based on the CLI arguments.
    pub fn ssl_config(&self) -> Option<SslClientConfig> {
        let vision = match self.connection_mode {
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
        let gc = match self.connection_mode {
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

impl ConnectionMode {
    /// CLI value string for forwarding to a child process.
    pub fn to_arg(&self) -> &'static str {
        match self {
            ConnectionMode::None => "none",
            ConnectionMode::Tcp => "tcp",
            ConnectionMode::Udp => "udp",
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
    pub async fn select(&self) -> Result<String> {
        select_serial_port(self).await
    }

    /// CLI value string for forwarding to a child process (inverse of `FromStr`).
    pub fn to_arg(&self) -> String {
        match self {
            SerialPort::Disabled => "disabled".to_owned(),
            SerialPort::Auto => "auto".to_owned(),
            SerialPort::Port(p) => p.clone(),
        }
    }
}

/// Parse `KEY=VALUE` strategy-param specs into a [`StrategyParams`] map. The
/// value is coerced to a bool (`true`/`false`), else an integer, else a float,
/// else kept as a string.
pub fn parse_strategy_params(specs: &[String]) -> Result<StrategyParams> {
    let mut out = StrategyParams::new();
    for spec in specs {
        let (key, value) = spec.split_once('=').ok_or_else(|| {
            anyhow::anyhow!("Invalid --strategy-param `{spec}` (expected KEY=VALUE)")
        })?;
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() {
            bail!("Invalid --strategy-param `{spec}` (empty key)");
        }
        let parsed = match value {
            "true" => ParamValue::Bool(true),
            "false" => ParamValue::Bool(false),
            _ if value.parse::<i32>().is_ok() => ParamValue::Int(value.parse().unwrap()),
            _ if value.parse::<f64>().is_ok() => ParamValue::Float(value.parse().unwrap()),
            _ => ParamValue::Text(value.to_owned()),
        };
        out.insert(key.to_owned(), parsed);
    }
    Ok(out)
}

impl FromStr for SerialPort {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        if s == "disabled" {
            Ok(SerialPort::Disabled)
        } else if s == "auto" {
            Ok(SerialPort::Auto)
        } else {
            Ok(SerialPort::Port(s.to_owned()))
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
                // Auto-pick a port. On macOS the same device shows up as both
                // a `cu.*` (call-up) and a `tty.*` entry; prefer `cu.*` since
                // that's the one meant for outgoing connections. Otherwise just
                // take the first port.
                let chosen = ports
                    .iter()
                    .find(|p| p.contains("cu."))
                    .unwrap_or(&ports[0])
                    .clone();
                log::info!(
                    "Auto-selected serial port {} (available: {})",
                    chosen,
                    ports.join(", ")
                );
                Some(chosen)
            }
        }
        SerialPort::Port(port) => {
            // An explicitly requested port is always honored — we only validate
            // that the device exists, not that autodetect recognized it as a
            // basestation. Some basestation firmwares report a custom USB
            // VID/PID that the `is_basestation` heuristic doesn't match, so they
            // never show up in `list_serial_ports()` even though they work fine.
            if !ports.contains(port) {
                if !std::path::Path::new(port).exists() {
                    println!(
                        "Detected basestation ports:\n{}",
                        ports
                            .iter()
                            .map(|p| format!("  - {}", p))
                            .collect::<Vec<_>>()
                            .join("\n")
                    );
                    bail!("Port {} does not exist", port);
                }
                log::warn!(
                    "Port {} not detected as a basestation (custom USB VID/PID?), using it anyway",
                    port
                );
            }
            Some(port.clone())
        }
    };

    port.ok_or(anyhow::anyhow!("Serial port not found"))
}
