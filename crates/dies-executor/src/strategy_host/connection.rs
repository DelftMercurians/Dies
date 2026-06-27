//! Strategy connection management.
//!
//! Handles IPC with a single strategy process using Unix domain sockets.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant, SystemTime};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dies_core::{SideAssignment, TeamColor, TeamData};
use dies_strategy_protocol::{
    DebugEntry, HostMessage, ParamSpec, PassResult, PlayerId, SkillCommand, SkillStatus,
    StrategyConfig, StrategyMessage, StrategyParams,
};
use thiserror::Error;
use tracing::{debug, error, info, trace, warn};

use super::CoordinateTransformer;

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Timeout for receiving a response from strategy (realtime/UI mode). On
/// timeout the host silently applies no command this frame — fine for a live
/// best-effort loop, fatal for deterministic lockstep.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

/// Timeout for receiving a response in blocking (headless self-play) mode.
/// Generous so a slow-but-alive strategy still replies, but finite so a crashed
/// or deadlocked strategy aborts the match instead of hanging forever.
const BLOCKING_RECV_TIMEOUT: Duration = Duration::from_secs(5);

/// Timeout for sending a message to strategy.
const SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// Errors that can occur with strategy connections.
#[derive(Debug, Error)]
pub enum ConnectionError {
    #[error("failed to spawn strategy process: {0}")]
    Spawn(#[source] io::Error),

    #[error("failed to create socket: {0}")]
    Socket(#[source] io::Error),

    #[error("strategy did not connect within timeout")]
    ConnectTimeout,

    #[error("strategy did not respond within blocking timeout")]
    Timeout,

    #[error("failed to send message: {0}")]
    Send(#[source] io::Error),

    #[error("failed to receive message: {0}")]
    Receive(#[source] io::Error),

    #[error("failed to serialize message: {0}")]
    Serialize(#[source] bincode::Error),

    #[error("failed to deserialize message: {0}")]
    Deserialize(#[source] bincode::Error),

    #[error("connection closed")]
    Closed,

    #[error("message too large: {size} bytes (max {max})")]
    MessageTooLarge { size: u32, max: u32 },

    #[error("strategy process exited unexpectedly: {0}")]
    ProcessExited(String),

    #[error("strategy not ready")]
    NotReady,

    #[error("protocol error: {0}")]
    Protocol(String),
}

/// The current state of a strategy connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Not connected, no process running.
    Disconnected,
    /// Process is starting up, waiting for connection.
    Starting,
    /// Connected and ready to receive world updates.
    Ready,
    /// Connection lost, will attempt to reconnect.
    Reconnecting,
}

/// Output from processing a strategy's response.
#[derive(Debug, Default)]
pub struct StrategyOutput {
    /// Skill commands for each player (None = continue previous skill).
    pub skill_commands: HashMap<PlayerId, Option<SkillCommand>>,
    /// Debug visualization data.
    pub debug_data: Vec<DebugEntry>,
    /// Role names for players.
    pub player_roles: HashMap<PlayerId, String>,
}

/// Forward a strategy `Log` message to the appropriate tracing level.
fn log_strategy_message(level: dies_strategy_protocol::LogLevel, message: &str) {
    match level {
        dies_strategy_protocol::LogLevel::Trace => trace!("[Strategy] {}", message),
        dies_strategy_protocol::LogLevel::Debug => debug!("[Strategy] {}", message),
        dies_strategy_protocol::LogLevel::Info => info!("[Strategy] {}", message),
        dies_strategy_protocol::LogLevel::Warn => warn!("[Strategy] {}", message),
        dies_strategy_protocol::LogLevel::Error => error!("[Strategy] {}", message),
    }
}

/// Manages a connection to a single strategy process.
///
/// The connection handles:
/// - Spawning the strategy process
/// - Establishing the Unix socket connection
/// - Sending world updates and receiving skill commands
/// - Coordinate transformation
/// - Process health monitoring and restart
pub struct StrategyConnection {
    /// Team this connection is for.
    team_color: TeamColor,
    /// Path to the strategy binary.
    strategy_path: PathBuf,
    /// Socket path for IPC.
    socket_path: PathBuf,
    /// The Unix listener for accepting connections.
    listener: Option<UnixListener>,
    /// The connected stream.
    stream: Option<UnixStream>,
    /// The strategy process handle.
    process: Option<Child>,
    /// Coordinate transformer.
    transformer: CoordinateTransformer,
    /// Current connection state.
    state: ConnectionState,
    /// Configuration to send on connect.
    config: StrategyConfig,
    /// Parameters the strategy declared in its `Ready` message.
    declared_params: Vec<ParamSpec>,
    /// Last skill statuses reported.
    last_skill_statuses: HashMap<PlayerId, SkillStatus>,
    /// Read buffer.
    read_buffer: Vec<u8>,
    /// Dev-only: hot-reload the process when the binary on disk changes.
    hot_reload: bool,
    /// Last observed modification time of the strategy binary (hot-reload).
    binary_mtime: Option<SystemTime>,
    /// Last time we stat'd the binary, to throttle the mtime check (hot-reload).
    last_mtime_check: Option<Instant>,
    /// Blocking lockstep mode (headless self-play): wait up to
    /// `BLOCKING_RECV_TIMEOUT` for each reply and treat a timeout as a hard
    /// error rather than silently dropping the frame.
    blocking: bool,
}

impl StrategyConnection {
    /// Create a new strategy connection.
    ///
    /// This does not start the strategy process - call `start()` for that.
    pub fn new(
        team_color: TeamColor,
        strategy_path: PathBuf,
        side_assignment: SideAssignment,
        config: StrategyConfig,
        hot_reload: bool,
        blocking: bool,
    ) -> Self {
        // Create a unique socket path
        let socket_path = std::env::temp_dir().join(format!(
            "dies-strategy-{}-{}.sock",
            std::process::id(),
            match team_color {
                TeamColor::Blue => "blue",
                TeamColor::Yellow => "yellow",
            }
        ));

        Self {
            team_color,
            strategy_path,
            socket_path,
            listener: None,
            stream: None,
            process: None,
            transformer: CoordinateTransformer::new(team_color, side_assignment),
            state: ConnectionState::Disconnected,
            config,
            declared_params: Vec::new(),
            last_skill_statuses: HashMap::new(),
            read_buffer: Vec::with_capacity(64 * 1024),
            hot_reload,
            binary_mtime: None,
            last_mtime_check: None,
            blocking,
        }
    }

    /// Read the strategy binary's modification time, if available.
    fn read_binary_mtime(&self) -> Option<SystemTime> {
        std::fs::metadata(&self.strategy_path)
            .and_then(|m| m.modified())
            .ok()
    }

    /// Returns true if hot-reload is enabled and the strategy binary on disk has
    /// been modified since the process was started.
    ///
    /// Throttled to a few checks per second so we don't stat the binary on every
    /// frame. The caller is expected to restart the connection on `true`.
    pub fn binary_changed(&mut self) -> bool {
        if !self.hot_reload {
            return false;
        }

        let now = Instant::now();
        if let Some(last) = self.last_mtime_check {
            if now.duration_since(last) < Duration::from_millis(500) {
                return false;
            }
        }
        self.last_mtime_check = Some(now);

        match (self.read_binary_mtime(), self.binary_mtime) {
            (Some(current), Some(known)) => current != known,
            _ => false,
        }
    }

    /// Get the current connection state.
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Get the team color.
    pub fn team_color(&self) -> TeamColor {
        self.team_color
    }

    /// The parameters the strategy declared in its `Ready` message.
    pub fn declared_params(&self) -> &[ParamSpec] {
        &self.declared_params
    }

    /// Push updated runtime parameter values to the strategy. No-op if the
    /// connection is not yet ready.
    pub fn send_params(&mut self, params: &StrategyParams) -> Result<(), ConnectionError> {
        if self.state != ConnectionState::Ready {
            return Ok(());
        }
        self.send_message(&HostMessage::SetParams(params.clone()))
    }

    /// Update the side assignment.
    pub fn set_side_assignment(&mut self, side_assignment: SideAssignment) {
        self.transformer.set_side_assignment(side_assignment);
    }

    /// Start the strategy process and wait for connection.
    pub fn start(&mut self) -> Result<(), ConnectionError> {
        // Clean up any existing socket file
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        // Create the listener
        let listener = UnixListener::bind(&self.socket_path).map_err(ConnectionError::Socket)?;
        listener
            .set_nonblocking(true)
            .map_err(ConnectionError::Socket)?;
        self.listener = Some(listener);

        info!(
            "Starting strategy process: {:?} with socket {:?}",
            self.strategy_path, self.socket_path
        );

        // Spawn the strategy process
        let process = Command::new(&self.strategy_path)
            .arg("--socket")
            .arg(&self.socket_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(ConnectionError::Spawn)?;

        self.process = Some(process);
        self.state = ConnectionState::Starting;

        // Snapshot the binary mtime so hot-reload can detect future rebuilds.
        self.binary_mtime = self.read_binary_mtime();
        self.last_mtime_check = Some(Instant::now());

        Ok(())
    }

    /// Try to accept a connection from the strategy process.
    ///
    /// Should be called periodically when in `Starting` state.
    pub fn try_accept(&mut self) -> Result<bool, ConnectionError> {
        if self.state != ConnectionState::Starting {
            return Ok(false);
        }

        // If we already have a stream (accepted previously but timed out waiting
        // for Ready), try reading Ready from the existing stream first.
        if self.stream.is_some() {
            match self.receive_message()? {
                Some(StrategyMessage::Ready { params }) => {
                    self.declared_params = params;
                    self.state = ConnectionState::Ready;
                    return Ok(true);
                }
                Some(other) => {
                    warn!("Expected Ready message, got {:?}", other);
                    return Err(ConnectionError::Protocol(
                        "Expected Ready message".to_string(),
                    ));
                }
                None => {
                    // Still waiting — check if process died
                    return self.check_process_alive();
                }
            }
        }

        let listener = self.listener.as_ref().ok_or(ConnectionError::NotReady)?;

        match listener.accept() {
            Ok((stream, _)) => {
                info!("Strategy connected for team {:?}", self.team_color);
                // The listener is non-blocking and on some platforms (notably
                // macOS) the accepted stream inherits O_NONBLOCK. Realtime mode
                // relies on that: a read that would block returns `WouldBlock`,
                // which we treat as "no message this frame". But in blocking
                // (headless) lockstep mode a non-blocking socket is fatal — when
                // the free-running loop fills the send buffer, `write_all`
                // returns `WouldBlock` after writing only the length prefix,
                // orphaning it in the stream and desyncing the strategy's
                // framing (the strategy then reads a length prefix as a message
                // body). Force the socket blocking so writes apply backpressure
                // and always emit a complete frame.
                if self.blocking {
                    stream
                        .set_nonblocking(false)
                        .map_err(ConnectionError::Socket)?;
                }
                let recv_timeout = if self.blocking {
                    BLOCKING_RECV_TIMEOUT
                } else {
                    RECV_TIMEOUT
                };
                stream
                    .set_read_timeout(Some(recv_timeout))
                    .map_err(ConnectionError::Socket)?;
                stream
                    .set_write_timeout(Some(SEND_TIMEOUT))
                    .map_err(ConnectionError::Socket)?;
                self.stream = Some(stream);

                // Send init message
                self.send_message(&HostMessage::Init {
                    config: self.config.clone(),
                })?;

                // Wait for ready response
                match self.receive_message()? {
                    Some(StrategyMessage::Ready { params }) => {
                        self.declared_params = params;
                        self.state = ConnectionState::Ready;
                        Ok(true)
                    }
                    Some(other) => {
                        warn!("Expected Ready message, got {:?}", other);
                        Err(ConnectionError::Protocol(
                            "Expected Ready message".to_string(),
                        ))
                    }
                    None => {
                        // Ready not yet received — will retry on next call
                        Ok(false)
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => self.check_process_alive(),
            Err(e) => Err(ConnectionError::Socket(e)),
        }
    }

    /// Check if the strategy process is still alive. Returns Ok(false) if alive,
    /// or Err(ProcessExited) if the process has exited.
    fn check_process_alive(&mut self) -> Result<bool, ConnectionError> {
        if let Some(ref mut process) = self.process {
            match process.try_wait() {
                Ok(Some(status)) => {
                    self.state = ConnectionState::Disconnected;
                    Err(ConnectionError::ProcessExited(format!(
                        "Process exited with status: {}",
                        status
                    )))
                }
                Ok(None) => Ok(false),
                Err(e) => {
                    warn!("Failed to check process status: {}", e);
                    Ok(false)
                }
            }
        } else {
            Ok(false)
        }
    }

    /// Send a world update to the strategy and receive skill commands.
    ///
    /// Returns None if the strategy is not ready or if there was a timeout.
    pub fn update(
        &mut self,
        team_data: &TeamData,
        skill_statuses: HashMap<PlayerId, SkillStatus>,
        pass_results: HashMap<PlayerId, PassResult>,
    ) -> Result<Option<StrategyOutput>, ConnectionError> {
        if self.state != ConnectionState::Ready {
            return Ok(None);
        }

        self.last_skill_statuses = skill_statuses.clone();

        // Create world snapshot
        let world =
            self.transformer
                .create_world_snapshot(team_data, &skill_statuses, team_data.dt);

        // Transform pass-result positions to the team-relative frame.
        let pass_results = pass_results
            .into_iter()
            .map(|(id, r)| (id, self.transformer.transform_pass_result(r)))
            .collect();

        // Send world update
        let msg = HostMessage::WorldUpdate {
            world,
            skill_statuses,
            pass_results,
        };
        self.send_message(&msg)?;

        if self.blocking {
            // Lockstep mode: the runner sends any Log frames *then* the Output
            // for this tick, so keep reading until we get the Output. A timeout
            // means the strategy is dead/stuck — abort loudly instead of
            // silently dropping the frame (which would diverge the match).
            loop {
                match self.receive_message()? {
                    Some(StrategyMessage::Output {
                        skill_commands,
                        debug_data,
                        player_roles,
                    }) => {
                        return Ok(Some(self.build_output(
                            skill_commands,
                            debug_data,
                            player_roles,
                        )))
                    }
                    Some(StrategyMessage::Log { level, message }) => {
                        log_strategy_message(level, &message);
                    }
                    Some(StrategyMessage::Ready { .. }) => {
                        warn!("Received unexpected Ready message");
                    }
                    None => return Err(ConnectionError::Timeout),
                }
            }
        }

        // Receive response (best-effort, realtime/UI mode)
        match self.receive_message()? {
            Some(StrategyMessage::Output {
                skill_commands,
                debug_data,
                player_roles,
            }) => Ok(Some(self.build_output(
                skill_commands,
                debug_data,
                player_roles,
            ))),
            Some(StrategyMessage::Log { level, message }) => {
                // Log the message and try again
                log_strategy_message(level, &message);
                // Return empty output - we'll get the real output next frame
                Ok(None)
            }
            Some(StrategyMessage::Ready { .. }) => {
                warn!("Received unexpected Ready message");
                Ok(None)
            }
            None => {
                // Timeout - strategy might be slow
                Ok(None)
            }
        }
    }

    /// Transform a strategy `Output` payload from team-relative to world
    /// coordinates and wrap it in a [`StrategyOutput`].
    fn build_output(
        &self,
        skill_commands: HashMap<PlayerId, Option<SkillCommand>>,
        debug_data: Vec<DebugEntry>,
        player_roles: HashMap<PlayerId, String>,
    ) -> StrategyOutput {
        let skill_commands = skill_commands
            .into_iter()
            .map(|(id, cmd)| {
                let transformed = cmd.map(|c| self.transformer.transform_skill_command(&c));
                (id, transformed)
            })
            .collect();
        let debug_data = self.transformer.transform_debug_entries(debug_data);
        StrategyOutput {
            skill_commands,
            debug_data,
            player_roles,
        }
    }

    /// Request graceful shutdown of the strategy process.
    pub fn shutdown(&mut self) -> Result<(), ConnectionError> {
        if self.state == ConnectionState::Ready {
            let _ = self.send_message(&HostMessage::Shutdown);
        }

        // Give process a moment to exit gracefully
        if let Some(ref mut process) = self.process {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(2) {
                match process.try_wait() {
                    Ok(Some(_)) => break,
                    Ok(None) => std::thread::sleep(Duration::from_millis(50)),
                    Err(_) => break,
                }
            }
            // Force kill if still running
            let _ = process.kill();
        }

        self.cleanup();
        Ok(())
    }

    /// Check if the process is still alive.
    pub fn is_alive(&mut self) -> bool {
        if let Some(ref mut process) = self.process {
            match process.try_wait() {
                Ok(Some(_)) => false, // Exited
                Ok(None) => true,     // Still running
                Err(_) => false,      // Error, assume dead
            }
        } else {
            false
        }
    }

    /// Clean up resources.
    fn cleanup(&mut self) {
        self.stream = None;
        self.listener = None;
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
        }
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }
        self.state = ConnectionState::Disconnected;
    }

    /// Send a message to the strategy.
    fn send_message(&mut self, msg: &HostMessage) -> Result<(), ConnectionError> {
        let data = bincode::serialize(msg).map_err(ConnectionError::Serialize)?;
        let len = data.len() as u32;

        if len > MAX_MESSAGE_SIZE {
            return Err(ConnectionError::MessageTooLarge {
                size: len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        let stream = self.stream.as_mut().ok_or(ConnectionError::NotReady)?;

        stream
            .write_u32::<LittleEndian>(len)
            .map_err(Self::convert_io_error)?;
        stream.write_all(&data).map_err(Self::convert_io_error)?;
        stream.flush().map_err(Self::convert_io_error)?;

        Ok(())
    }

    /// Convert an IO error to a ConnectionError (static version).
    fn convert_io_error(error: io::Error) -> ConnectionError {
        if error.kind() == io::ErrorKind::BrokenPipe
            || error.kind() == io::ErrorKind::ConnectionReset
            || error.kind() == io::ErrorKind::UnexpectedEof
        {
            ConnectionError::Closed
        } else {
            ConnectionError::Send(error)
        }
    }

    /// Receive a message from the strategy.
    fn receive_message(&mut self) -> Result<Option<StrategyMessage>, ConnectionError> {
        let stream = self.stream.as_mut().ok_or(ConnectionError::NotReady)?;

        // Read length prefix
        let len = match stream.read_u32::<LittleEndian>() {
            Ok(len) => len,
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
            Err(e) if e.kind() == io::ErrorKind::TimedOut => return Ok(None),
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                self.state = ConnectionState::Reconnecting;
                return Err(ConnectionError::Closed);
            }
            Err(e) => return Err(self.handle_io_error(e)),
        };

        if len > MAX_MESSAGE_SIZE {
            return Err(ConnectionError::MessageTooLarge {
                size: len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        // Read message data
        let len = len as usize;
        if self.read_buffer.len() < len {
            self.read_buffer.resize(len, 0);
        }

        stream
            .read_exact(&mut self.read_buffer[..len])
            .map_err(|e| self.handle_io_error(e))?;

        let msg: StrategyMessage =
            bincode::deserialize(&self.read_buffer[..len]).map_err(ConnectionError::Deserialize)?;

        Ok(Some(msg))
    }

    /// Handle an IO error, updating state if connection is lost.
    fn handle_io_error(&mut self, error: io::Error) -> ConnectionError {
        if error.kind() == io::ErrorKind::BrokenPipe
            || error.kind() == io::ErrorKind::ConnectionReset
            || error.kind() == io::ErrorKind::UnexpectedEof
        {
            self.state = ConnectionState::Reconnecting;
            ConnectionError::Closed
        } else {
            ConnectionError::Receive(error)
        }
    }
}

impl Drop for StrategyConnection {
    fn drop(&mut self) {
        self.cleanup();
    }
}
