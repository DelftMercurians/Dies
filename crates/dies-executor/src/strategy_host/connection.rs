//! Strategy connection management.
//!
//! Handles IPC with a single strategy process using Unix domain sockets.

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dies_core::{SideAssignment, TeamColor, TeamData};
use dies_strategy_protocol::{
    DebugEntry, HostMessage, PlayerId, SkillCommand, SkillStatus, StrategyConfig, StrategyMessage,
};
use thiserror::Error;
use tracing::{debug, error, info, trace, warn};

use super::CoordinateTransformer;

/// Maximum message size (16 MB).
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Timeout for receiving a response from strategy.
const RECV_TIMEOUT: Duration = Duration::from_millis(100);

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
    /// Last skill statuses reported.
    last_skill_statuses: HashMap<PlayerId, SkillStatus>,
    /// Read buffer.
    read_buffer: Vec<u8>,
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
            last_skill_statuses: HashMap::new(),
            read_buffer: Vec::with_capacity(64 * 1024),
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

        Ok(())
    }

    /// Try to accept a connection from the strategy process.
    ///
    /// Should be called periodically when in `Starting` state.
    pub fn try_accept(&mut self) -> Result<bool, ConnectionError> {
        if self.state != ConnectionState::Starting {
            return Ok(false);
        }

        let listener = self.listener.as_ref().ok_or(ConnectionError::NotReady)?;

        match listener.accept() {
            Ok((stream, _)) => {
                info!("Strategy connected for team {:?}", self.team_color);
                stream
                    .set_read_timeout(Some(RECV_TIMEOUT))
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
                    Some(StrategyMessage::Ready) => {
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
                        // Try again later
                        Ok(false)
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No connection yet, check if process is still alive
                if let Some(ref mut process) = self.process {
                    match process.try_wait() {
                        Ok(Some(status)) => {
                            self.state = ConnectionState::Disconnected;
                            return Err(ConnectionError::ProcessExited(format!(
                                "Process exited with status: {}",
                                status
                            )));
                        }
                        Ok(None) => {
                            // Still running
                        }
                        Err(e) => {
                            warn!("Failed to check process status: {}", e);
                        }
                    }
                }
                Ok(false)
            }
            Err(e) => Err(ConnectionError::Socket(e)),
        }
    }

    /// Send a world update to the strategy and receive skill commands.
    ///
    /// Returns None if the strategy is not ready or if there was a timeout.
    pub fn update(
        &mut self,
        team_data: &TeamData,
        skill_statuses: HashMap<PlayerId, SkillStatus>,
    ) -> Result<Option<StrategyOutput>, ConnectionError> {
        if self.state != ConnectionState::Ready {
            return Ok(None);
        }

        self.last_skill_statuses = skill_statuses.clone();

        // Create world snapshot
        let world = self.transformer.create_world_snapshot(
            team_data,
            &skill_statuses,
            team_data.dt,
        );

        // Send world update
        let msg = HostMessage::WorldUpdate {
            world,
            skill_statuses,
        };
        self.send_message(&msg)?;

        // Receive response
        match self.receive_message()? {
            Some(StrategyMessage::Output {
                skill_commands,
                debug_data,
                player_roles,
            }) => {
                // Transform skill commands from strategy to world coordinates
                let transformed_commands: HashMap<PlayerId, Option<SkillCommand>> = skill_commands
                    .into_iter()
                    .map(|(id, cmd)| {
                        let transformed = cmd.map(|c| self.transformer.transform_skill_command(&c));
                        (id, transformed)
                    })
                    .collect();

                // Transform debug data
                let transformed_debug = self.transformer.transform_debug_entries(debug_data);

                Ok(Some(StrategyOutput {
                    skill_commands: transformed_commands,
                    debug_data: transformed_debug,
                    player_roles,
                }))
            }
            Some(StrategyMessage::Log { level, message }) => {
                // Log the message and try again
                match level {
                    dies_strategy_protocol::LogLevel::Trace => trace!("[Strategy] {}", message),
                    dies_strategy_protocol::LogLevel::Debug => debug!("[Strategy] {}", message),
                    dies_strategy_protocol::LogLevel::Info => info!("[Strategy] {}", message),
                    dies_strategy_protocol::LogLevel::Warn => warn!("[Strategy] {}", message),
                    dies_strategy_protocol::LogLevel::Error => error!("[Strategy] {}", message),
                }
                // Return empty output - we'll get the real output next frame
                Ok(None)
            }
            Some(StrategyMessage::Ready) => {
                warn!("Received unexpected Ready message");
                Ok(None)
            }
            None => {
                // Timeout - strategy might be slow
                Ok(None)
            }
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
        stream
            .write_all(&data)
            .map_err(Self::convert_io_error)?;
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

