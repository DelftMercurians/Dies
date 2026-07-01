//! Main strategy runner loop.
//!
//! The runner handles:
//! - Command-line argument parsing
//! - Connection establishment
//! - Main update loop
//! - Graceful shutdown

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use dies_strategy_api::{debug as strategy_debug, Strategy, TeamContext, World};
use dies_strategy_protocol::{HostMessage, StrategyConfig, StrategyMessage, StrategyParams};
use tracing::{debug, error, info, warn};

use crate::ipc::{Connection, ConnectionError};
use crate::logging::{self, LogEntry};

/// Command-line arguments for strategy processes.
#[derive(Parser, Debug, Clone)]
#[command(author, version, about = "Dies strategy process")]
pub struct RunnerArgs {
    /// Path to the Unix domain socket for host communication.
    #[arg(long, env = "DIES_STRATEGY_SOCKET")]
    pub socket: PathBuf,

    /// Optional configuration JSON string.
    #[arg(long, env = "DIES_STRATEGY_CONFIG")]
    pub config: Option<String>,

    /// Enable verbose logging.
    #[arg(short, long)]
    pub verbose: bool,
}

/// Strategy runner that manages the main loop and IPC.
pub struct StrategyRunner<S: Strategy> {
    strategy: S,
    connection: Connection,
    config: StrategyConfig,
    log_receiver: Receiver<LogEntry>,
    running: bool,
    frame_count: u64,
    /// Current runtime parameter values (declared defaults + UI overrides).
    params: StrategyParams,
    /// Whether `strategy.init` has been called (on the first world update).
    initialized: bool,
}

impl<S: Strategy> StrategyRunner<S> {
    /// Create a new runner with an established connection and strategy.
    fn new(
        strategy: S,
        connection: Connection,
        config: StrategyConfig,
        log_receiver: Receiver<LogEntry>,
        params: StrategyParams,
    ) -> Self {
        Self {
            strategy,
            connection,
            config,
            log_receiver,
            running: true,
            frame_count: 0,
            params,
            initialized: false,
        }
    }

    /// Run the main loop until shutdown.
    fn run(&mut self) -> Result<()> {
        info!("strategy runner starting main loop");

        while self.running {
            match self.process_frame() {
                Ok(()) => {}
                Err(e) => {
                    // Check if it's a clean shutdown
                    if let Some(ConnectionError::Closed) = e.downcast_ref::<ConnectionError>() {
                        info!("connection closed by host, shutting down");
                        break;
                    }
                    // A read timeout at a frame boundary just means the host is
                    // idle (e.g. the executor is paused). The stream is still
                    // aligned, so keep waiting instead of tearing down the
                    // strategy — otherwise resuming from a long pause drops the
                    // connection.
                    if let Some(ConnectionError::Timeout) = e.downcast_ref::<ConnectionError>() {
                        debug!("no frame from host within read timeout; still waiting");
                        continue;
                    }
                    error!("error processing frame: {:#}", e);
                    return Err(e);
                }
            }
        }

        // Call strategy shutdown
        info!("calling strategy shutdown");
        self.strategy.shutdown();

        Ok(())
    }

    /// Process a single frame.
    fn process_frame(&mut self) -> Result<()> {
        // Receive message from host
        let message = self
            .connection
            .receive()
            .context("failed to receive message from host")?;

        match message {
            HostMessage::Init { config } => {
                warn!("received unexpected Init message after initialization");
                self.config = config;
            }

            HostMessage::SetParams(params) => {
                // Merge overrides on top of current values (keeps declared defaults
                // for keys the UI hasn't touched).
                for (key, value) in params {
                    self.params.insert(key, value);
                }
            }

            HostMessage::WorldUpdate {
                world,
                skill_statuses,
                pass_results,
            } => {
                let frame_start = Instant::now();
                self.frame_count += 1;

                // Lazily call init on the first world update.
                if !self.initialized {
                    let init_world = World::new(world.clone());
                    self.strategy.init(&init_world);
                    self.initialized = true;
                }

                // Create TeamContext from world snapshot + current params
                let mut ctx =
                    TeamContext::new(world, skill_statuses, pass_results, self.params.clone());

                // Call strategy update
                self.strategy.update(&mut ctx);

                // Collect outputs
                let (skill_commands, player_roles) = ctx.collect_output();
                let debug_data = strategy_debug::collect_entries();

                // Collect logs
                let logs = logging::collect_logs(&self.log_receiver);

                // Send logs first (if any)
                for log_msg in logging::logs_to_messages(logs) {
                    if let Err(e) = self.connection.send(&log_msg) {
                        warn!("failed to send log message: {}", e);
                    }
                }

                // Send output
                let output = StrategyMessage::Output {
                    skill_commands,
                    debug_data,
                    player_roles,
                };

                self.connection
                    .send(&output)
                    .context("failed to send output to host")?;

                let frame_time = frame_start.elapsed();
                if frame_time > Duration::from_millis(10) {
                    debug!("frame {} took {:?} (> 10ms)", self.frame_count, frame_time);
                }
            }

            HostMessage::Shutdown => {
                info!("received shutdown message from host");
                self.running = false;
            }
        }

        Ok(())
    }

    /// Get the current frame count.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

/// Run a strategy with command-line arguments.
///
/// This is the main entry point for strategy binaries. It:
/// 1. Parses command-line arguments
/// 2. Initializes logging
/// 3. Connects to the host
/// 4. Runs the main loop
/// 5. Handles graceful shutdown
///
/// # Arguments
///
/// * `factory` - Factory function that creates the strategy instance
///
/// # Example
///
/// ```ignore
/// use dies_strategy_api::prelude::*;
/// use dies_strategy_runner::run_strategy;
///
/// struct MyStrategy;
///
/// impl Strategy for MyStrategy {
///     fn update(&mut self, ctx: &mut TeamContext) {
///         // Strategy logic
///     }
/// }
///
/// fn main() {
///     run_strategy(|| MyStrategy);
/// }
/// ```
pub fn run_strategy<S, F>(factory: F)
where
    S: Strategy,
    F: FnOnce() -> S,
{
    let args = RunnerArgs::parse();
    if let Err(e) = run_strategy_with_args(factory, args) {
        error!("strategy runner failed: {:#}", e);
        std::process::exit(1);
    }
}

/// Run a strategy with explicit arguments.
///
/// Useful for testing or when you need more control over initialization.
pub fn run_strategy_with_args<S, F>(factory: F, args: RunnerArgs) -> Result<()>
where
    S: Strategy,
    F: FnOnce() -> S,
{
    // Initialize logging
    let log_receiver = logging::init_logging();

    info!("dies strategy runner starting");
    info!("socket: {}", args.socket.display());

    // Connect to host
    let mut connection = Connection::connect(&args.socket)
        .with_context(|| format!("failed to connect to {}", args.socket.display()))?;

    // Wait for Init message
    let config = match connection
        .receive()
        .context("failed to receive Init message")?
    {
        HostMessage::Init { config } => {
            info!("received Init from host");
            config
        }
        HostMessage::Shutdown => {
            info!("received Shutdown before Init, exiting");
            return Ok(());
        }
        other => {
            anyhow::bail!(
                "expected Init message, got {:?}",
                std::mem::discriminant(&other)
            );
        }
    };

    // Create the strategy so we can advertise its declared parameters.
    let strategy = factory();
    let specs = strategy.params();

    // Seed current values with the declared defaults; UI overrides arrive later
    // via `HostMessage::SetParams` (including the host's cached re-send).
    let params: StrategyParams = specs
        .iter()
        .map(|s| (s.key.clone(), s.default.clone()))
        .collect();

    // Send Ready, advertising the parameters to the host/UI.
    connection
        .send(&StrategyMessage::Ready {
            params: specs.clone(),
        })
        .context("failed to send Ready message")?;
    info!("sent Ready to host ({} params)", specs.len());

    // Create runner and run the main loop. `init` and the first frame are handled
    // lazily inside the loop, so params pushed before the first WorldUpdate apply.
    let mut runner = StrategyRunner::new(strategy, connection, config, log_receiver, params);

    // Set up signal handling for graceful shutdown
    setup_signal_handler()?;

    runner.run()?;

    info!("strategy runner exiting cleanly");
    Ok(())
}

/// Set up signal handlers for graceful shutdown.
fn setup_signal_handler() -> Result<()> {
    // Signal handling is done via tokio in async context
    // For sync context, we rely on the host sending Shutdown message
    // or the socket being closed
    debug!("signal handling deferred to host");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{GameState, WorldSnapshot};
    use std::collections::HashMap;

    struct TestStrategy {
        update_count: u32,
    }

    impl TestStrategy {
        fn new() -> Self {
            Self { update_count: 0 }
        }
    }

    impl Strategy for TestStrategy {
        fn update(&mut self, _ctx: &mut TeamContext) {
            self.update_count += 1;
        }
    }

    #[test]
    fn test_runner_args_parsing() {
        let args = RunnerArgs::try_parse_from([
            "test",
            "--socket",
            "/tmp/test.sock",
            "--config",
            "{}",
            "-v",
        ])
        .unwrap();

        assert_eq!(args.socket, PathBuf::from("/tmp/test.sock"));
        assert_eq!(args.config, Some("{}".to_string()));
        assert!(args.verbose);
    }

    #[test]
    fn test_team_context_creation() {
        let snapshot = WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: None,
            ball: None,
            own_players: vec![],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            pre_stage: false,
            our_keeper_id: None,
            freekick_kicker: None,
            possession: dies_strategy_protocol::Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        };

        let ctx = TeamContext::new(snapshot, HashMap::new(), HashMap::new(), HashMap::new());

        assert_eq!(ctx.player_count(), 0);
    }
}
