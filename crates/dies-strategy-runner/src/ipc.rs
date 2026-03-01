//! IPC client for strategy-host communication.
//!
//! Handles Unix domain socket connections with length-prefixed binary framing.

use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::time::Duration;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use dies_strategy_protocol::{HostMessage, StrategyMessage};
use thiserror::Error;
use tracing::{debug, trace, warn};

/// Errors that can occur during IPC operations.
#[derive(Debug, Error)]
pub enum ConnectionError {
    /// Failed to connect to the host socket.
    #[error("failed to connect to socket: {0}")]
    Connect(#[source] io::Error),

    /// Failed to send a message.
    #[error("failed to send message: {0}")]
    Send(#[source] io::Error),

    /// Failed to receive a message.
    #[error("failed to receive message: {0}")]
    Receive(#[source] io::Error),

    /// Failed to serialize a message.
    #[error("failed to serialize message: {0}")]
    Serialize(#[source] bincode::Error),

    /// Failed to deserialize a message.
    #[error("failed to deserialize message: {0}")]
    Deserialize(#[source] bincode::Error),

    /// Connection was closed by the host.
    #[error("connection closed by host")]
    Closed,

    /// Message too large.
    #[error("message too large: {size} bytes (max {max})")]
    MessageTooLarge { size: u32, max: u32 },

    /// Connection timeout.
    #[error("connection timeout")]
    Timeout,
}

/// Maximum message size (16 MB should be plenty for world updates).
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Default read timeout for blocking reads.
const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// Default write timeout for blocking writes.
const DEFAULT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// Unix socket connection to the strategy host.
///
/// Uses length-prefixed framing:
/// - 4 bytes: message length (little-endian u32)
/// - N bytes: bincode-serialized message
pub struct Connection {
    stream: UnixStream,
    read_buffer: Vec<u8>,
}

impl Connection {
    /// Connect to a Unix domain socket at the given path.
    ///
    /// # Arguments
    ///
    /// * `socket_path` - Path to the Unix domain socket
    ///
    /// # Example
    ///
    /// ```ignore
    /// let conn = Connection::connect("/tmp/dies-strategy-12345.sock")?;
    /// ```
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self, ConnectionError> {
        let path = socket_path.as_ref();
        debug!("connecting to socket: {}", path.display());

        let stream = UnixStream::connect(path).map_err(ConnectionError::Connect)?;

        // Set timeouts for safety
        stream
            .set_read_timeout(Some(DEFAULT_READ_TIMEOUT))
            .map_err(ConnectionError::Connect)?;
        stream
            .set_write_timeout(Some(DEFAULT_WRITE_TIMEOUT))
            .map_err(ConnectionError::Connect)?;

        debug!("connected to host");

        Ok(Self {
            stream,
            read_buffer: Vec::with_capacity(64 * 1024), // Pre-allocate 64KB
        })
    }

    /// Set the read timeout.
    pub fn set_read_timeout(&self, timeout: Option<Duration>) -> Result<(), ConnectionError> {
        self.stream
            .set_read_timeout(timeout)
            .map_err(ConnectionError::Connect)
    }

    /// Set the write timeout.
    pub fn set_write_timeout(&self, timeout: Option<Duration>) -> Result<(), ConnectionError> {
        self.stream
            .set_write_timeout(timeout)
            .map_err(ConnectionError::Connect)
    }

    /// Send a message to the host.
    ///
    /// Messages are serialized with bincode and framed with a 4-byte length prefix.
    pub fn send(&mut self, message: &StrategyMessage) -> Result<(), ConnectionError> {
        trace!("sending message: {:?}", std::mem::discriminant(message));

        // Serialize the message
        let data = bincode::serialize(message).map_err(ConnectionError::Serialize)?;

        let len = data.len() as u32;
        if len > MAX_MESSAGE_SIZE {
            return Err(ConnectionError::MessageTooLarge {
                size: len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        // Write length prefix
        self.stream
            .write_u32::<LittleEndian>(len)
            .map_err(ConnectionError::Send)?;

        // Write message data
        self.stream
            .write_all(&data)
            .map_err(ConnectionError::Send)?;

        // Flush to ensure the message is sent
        self.stream.flush().map_err(ConnectionError::Send)?;

        trace!("sent {} bytes", len + 4);
        Ok(())
    }

    /// Receive a message from the host.
    ///
    /// Blocks until a complete message is received.
    pub fn receive(&mut self) -> Result<HostMessage, ConnectionError> {
        trace!("waiting for message");

        // Read length prefix
        let len = match self.stream.read_u32::<LittleEndian>() {
            Ok(len) => len,
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => {
                debug!("connection closed by host (EOF)");
                return Err(ConnectionError::Closed);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Err(ConnectionError::Timeout);
            }
            Err(e) => return Err(ConnectionError::Receive(e)),
        };

        if len > MAX_MESSAGE_SIZE {
            warn!("received oversized message: {} bytes", len);
            return Err(ConnectionError::MessageTooLarge {
                size: len,
                max: MAX_MESSAGE_SIZE,
            });
        }

        // Resize read buffer if needed
        let len = len as usize;
        if self.read_buffer.len() < len {
            self.read_buffer.resize(len, 0);
        }

        // Read message data
        self.stream
            .read_exact(&mut self.read_buffer[..len])
            .map_err(|e| {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    ConnectionError::Closed
                } else {
                    ConnectionError::Receive(e)
                }
            })?;

        // Deserialize
        let message: HostMessage =
            bincode::deserialize(&self.read_buffer[..len]).map_err(ConnectionError::Deserialize)?;

        trace!(
            "received {} bytes: {:?}",
            len + 4,
            std::mem::discriminant(&message)
        );
        Ok(message)
    }

    /// Try to receive a message without blocking.
    ///
    /// Returns `Ok(None)` if no message is available.
    pub fn try_receive(&mut self) -> Result<Option<HostMessage>, ConnectionError> {
        // Temporarily set non-blocking mode
        self.stream
            .set_nonblocking(true)
            .map_err(ConnectionError::Receive)?;

        let result = self.receive();

        // Restore blocking mode
        self.stream
            .set_nonblocking(false)
            .map_err(ConnectionError::Receive)?;

        match result {
            Ok(msg) => Ok(Some(msg)),
            Err(ConnectionError::Timeout) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Close the connection.
    pub fn close(self) {
        // Stream is closed when dropped
        debug!("closing connection");
        drop(self.stream);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{
        GameState, SkillCommand, SkillStatus, StrategyConfig, WorldSnapshot,
    };
    use std::collections::HashMap;
    use std::os::unix::net::UnixListener;
    use std::thread;
    use tempfile::tempdir;

    fn create_test_socket() -> (std::path::PathBuf, UnixListener) {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.sock");
        let listener = UnixListener::bind(&path).unwrap();
        // Leak the tempdir so the socket file persists for the test
        std::mem::forget(dir);
        (path, listener)
    }

    #[test]
    fn test_connection_roundtrip() {
        let (path, listener) = create_test_socket();

        // Spawn server thread
        let server_handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();

            // Read length prefix
            let len = stream.read_u32::<LittleEndian>().unwrap();

            // Read message
            let mut buf = vec![0u8; len as usize];
            stream.read_exact(&mut buf).unwrap();

            let msg: StrategyMessage = bincode::deserialize(&buf).unwrap();
            assert!(matches!(msg, StrategyMessage::Ready));

            // Send response
            let response = HostMessage::Init {
                config: StrategyConfig::default(),
            };
            let data = bincode::serialize(&response).unwrap();
            stream.write_u32::<LittleEndian>(data.len() as u32).unwrap();
            stream.write_all(&data).unwrap();
        });

        // Client side
        let mut conn = Connection::connect(&path).unwrap();
        conn.send(&StrategyMessage::Ready).unwrap();
        let response = conn.receive().unwrap();

        assert!(matches!(response, HostMessage::Init { .. }));

        server_handle.join().unwrap();
    }

    #[test]
    fn test_world_update_roundtrip() {
        let (path, listener) = create_test_socket();

        let server_handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();

            // Send world update
            let world = WorldSnapshot {
                timestamp: 12345.678,
                dt: 0.016,
                field_geom: None,
                ball: None,
                own_players: vec![],
                opp_players: vec![],
                game_state: GameState::Run,
                us_operating: true,
                our_keeper_id: None,
                freekick_kicker: None,
            };

            let msg = HostMessage::WorldUpdate {
                world,
                skill_statuses: HashMap::new(),
            };

            let data = bincode::serialize(&msg).unwrap();
            stream.write_u32::<LittleEndian>(data.len() as u32).unwrap();
            stream.write_all(&data).unwrap();
        });

        let mut conn = Connection::connect(&path).unwrap();
        let msg = conn.receive().unwrap();

        match msg {
            HostMessage::WorldUpdate { world, .. } => {
                assert!((world.timestamp - 12345.678).abs() < 1e-6);
                assert!((world.dt - 0.016).abs() < 1e-6);
                assert_eq!(world.game_state, GameState::Run);
            }
            _ => panic!("Expected WorldUpdate"),
        }

        server_handle.join().unwrap();
    }

    #[test]
    fn test_skill_commands_roundtrip() {
        let (path, listener) = create_test_socket();

        let server_handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();

            // Read skill command message
            let len = stream.read_u32::<LittleEndian>().unwrap();
            let mut buf = vec![0u8; len as usize];
            stream.read_exact(&mut buf).unwrap();

            let msg: StrategyMessage = bincode::deserialize(&buf).unwrap();
            match msg {
                StrategyMessage::Output { skill_commands, .. } => {
                    assert!(!skill_commands.is_empty());
                    let cmd = skill_commands
                        .get(&dies_strategy_protocol::PlayerId::new(1))
                        .unwrap();
                    assert!(matches!(cmd, Some(SkillCommand::GoToPos { .. })));
                }
                _ => panic!("Expected Output"),
            }
        });

        let mut conn = Connection::connect(&path).unwrap();

        let mut skill_commands = HashMap::new();
        skill_commands.insert(
            dies_strategy_protocol::PlayerId::new(1),
            Some(SkillCommand::GoToPos {
                position: dies_strategy_protocol::Vector2::new(1000.0, 500.0),
                heading: None,
            }),
        );

        let msg = StrategyMessage::Output {
            skill_commands,
            debug_data: vec![],
            player_roles: HashMap::new(),
        };

        conn.send(&msg).unwrap();

        server_handle.join().unwrap();
    }
}
