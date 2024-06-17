use std::{collections::HashMap, io::Write, time::Duration};

use anyhow::{Context, Result};
use serialport::available_ports;
use tokio::sync::{mpsc, oneshot};

use dies_core::{PlayerCmd, PlayerFeedbackMsg, PlayerId};

const MAX_MSG_FREQ: f64 = 60.0;
const TIMEOUT: Duration = Duration::from_millis(20);

/// List available serial ports. The port names can be used to create a
/// [`SerialClient`].
pub fn list_serial_ports() -> Result<Vec<String>> {
    available_ports()
        .context("Failed to get available ports")?
        .iter()
        .map(|p| Ok(p.port_name.to_string()))
        .collect()
}

/// Configuration for the serial client.
#[derive(Debug, Clone)]
pub struct SerialClientConfig {
    /// The name of the serial port. Use [`list_serial_ports`] to get a list of
    /// available ports.
    port_name: String,
    /// The baud rate of the serial port. The default is 115200.
    baud_rate: u32,
    /// Map of player IDs to robot ids
    robot_id_map: HashMap<PlayerId, u32>,
}

impl SerialClientConfig {
    pub fn new(port: String) -> Self {
        SerialClientConfig {
            port_name: port,
            baud_rate: 115200,
            robot_id_map: HashMap::new(),
        }
    }

    pub fn new_with_id_map(port: String, robot_id_map: HashMap<PlayerId, u32>) -> Self {
        SerialClientConfig {
            port_name: port,
            baud_rate: 115200,
            robot_id_map,
        }
    }

    pub fn set_robot_id_map_from_string(&mut self, map: &str) {
        // Parse string "<id1>:<id1>;..."
        self.robot_id_map = map
            .split(';')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let mut parts = s.split(':');
                let player_id = parts
                    .next()
                    .expect("Failed to parse player id")
                    .parse::<u32>()
                    .expect("Failed to parse player id");
                let robot_id = parts
                    .next()
                    .expect("Failed to parse robot id")
                    .parse::<u32>()
                    .expect("Failed to parse robot id");
                (PlayerId::new(player_id), robot_id)
            })
            .collect();
    }
}

impl Default for SerialClientConfig {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            port_name: "COM3".to_string(),
            #[cfg(not(target_os = "windows"))]
            port_name: "/dev/ttyACM0".to_string(),
            baud_rate: 115200,
            robot_id_map: HashMap::new(),
        }
    }
}

/// Async client for the serial port.
pub struct SerialClient {
    writer_tx: mpsc::UnboundedSender<(PlayerCmd, oneshot::Sender<Result<()>>)>,
}

impl SerialClient {
    /// Create a new `SerialClient`.
    pub fn new(config: SerialClientConfig) -> Result<Self> {
        let SerialClientConfig {
            port_name,
            baud_rate,
            robot_id_map,
        } = config;

        let mut port = serialport::new(port_name, baud_rate)
        .timeout(TIMEOUT)
        .open()
        .map_err(|err| {
            if let serialport::ErrorKind::Io(kind) = &err.kind {
                if kind == &std::io::ErrorKind::PermissionDenied {
                    anyhow::anyhow!(r#"Permission denied. If you are on Linux, you may need to add your user to the dialout group. For Debian based systems, see (https://askubuntu.com/questions/210177/serial-port-terminal-cannot-open-dev-ttys0-permission-denied).For Arch based systems, see (https://github.com/esp8266/source-code-examples/issues/26#issuecomment-320999460)."#)
                } else {
                    err.into()
                }
            } else {
                err.into()
            }
        })
        .context("Failed to open serial port")?;

        port.set_timeout(TIMEOUT)
            .context("Failed to set serial port timeout")?;

        // Launch a blocking thread for writing to the serial port
        let (tx, mut rx) = mpsc::unbounded_channel::<(PlayerCmd, oneshot::Sender<Result<()>>)>();
        tokio::task::spawn_blocking(move || {
            let mut last_time = std::time::Instant::now();

            loop {
                match rx.blocking_recv() {
                    Some((msg, sender)) => {
                        let elapsed = last_time.elapsed().as_secs_f64();
                        if elapsed < 1.0 / MAX_MSG_FREQ {
                            log::warn!("Message frequency too high, skipping message");
                            continue;
                        }

                        let mut msg = msg;
                        msg.id = robot_id_map
                            .get(&msg.id)
                            .map(|id| PlayerId::new(*id))
                            .unwrap_or(msg.id);

                        let cmd = msg.to_string();
                        // Write to serial port with timeout
                        let write_res = {
                            let mut buf = cmd.as_bytes();
                            let start = std::time::Instant::now();
                            loop {
                                if buf.is_empty() {
                                    break Ok(());
                                }
                                if start.elapsed() > TIMEOUT {
                                    break Err(anyhow::anyhow!("Timeout writing to serial port"));
                                }

                                match port.write(buf) {
                                    Ok(0) => {
                                        break Err(anyhow::anyhow!(
                                            "Failed to write to serial port"
                                        ));
                                    }
                                    Ok(n) => buf = &buf[n..],
                                    Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                                    Err(e) => {
                                        break Err(e.into());
                                    }
                                }
                            }
                        };

                        // Check if there was an error writing to the serial port
                        if let Err(err) = write_res {
                            log::error!("Error writing to serial port: {}", err);
                            sender.send(Err(err.into())).ok();
                        } else {
                            sender.send(Ok(())).ok();
                        }
                        last_time = std::time::Instant::now();
                    }
                    None => break,
                }
            }
            port.clear(serialport::ClearBuffer::All).unwrap();
            drop(port);
            println!("Closing serial port");
        });

        Ok(Self { writer_tx: tx })
    }

    /// Send a message to the serial port.
    pub async fn send(&mut self, msg: PlayerCmd) -> Result<()> {
        std::io::stdout().flush().unwrap();
        let (tx, rx) = oneshot::channel();
        self.writer_tx
            .send((msg, tx))
            .context("Failed to send message to serial port")?;
        rx.await?
    }

    pub fn send_no_wait(&mut self, msg: PlayerCmd) {
        let (tx, _) = oneshot::channel();
        self.writer_tx.send((msg, tx)).ok();
    }

    /// Receive a message from the serial port.
    pub async fn recv(&mut self) -> Result<PlayerFeedbackMsg> {
        Err(anyhow::anyhow!("Not implemented"))
    }
}
