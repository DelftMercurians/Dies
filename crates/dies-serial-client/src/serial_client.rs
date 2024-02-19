use anyhow::Result;
use serialport::available_ports;
use tokio::sync::{mpsc, oneshot};

use dies_core::PlayerCmd;

/// List available serial ports. The port names can be used to create a
/// [`SerialClient`].
pub fn list_serial_ports() -> Result<Vec<String>> {
    available_ports()?
        .iter()
        .map(|p| Ok(p.port_name.to_string()))
        .collect()
}

/// Configuration for the serial client.
pub struct SerialClientConfig {
    /// The name of the serial port. Use [`list_serial_ports`] to get a list of
    /// available ports.
    pub port_name: String,
    /// The baud rate of the serial port. The default is 115200.
    pub baud_rate: u32,
}

impl Default for SerialClientConfig {
    fn default() -> Self {
        Self {
            #[cfg(target_os = "windows")]
            port_name: "COM3".to_string(),
            #[cfg(not(target_os = "windows"))]
            port_name: "/dev/ttyACM0".to_string(),
            baud_rate: 115200,
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
        let mut port = serialport::new(config.port_name, config.baud_rate)
            .timeout(std::time::Duration::from_millis(10))
            .open()?;

        // Launch a blocking thread for writing to the serial port
        let (tx, mut rx) = mpsc::unbounded_channel::<(PlayerCmd, oneshot::Sender<Result<()>>)>();
        tokio::task::spawn_blocking(move || {
            let mut last_time = std::time::Instant::now();
            const TARGET_FREQ: f64 = 30.0;
            loop {
                match rx.blocking_recv() {
                    Some((msg, sender)) => {
                        port.clear(serialport::ClearBuffer::Input).unwrap();
                        // Limit the frequency of messages to the serial port
                        let elapsed = last_time.elapsed().as_secs_f64();
                        if elapsed < 1.0 / TARGET_FREQ {
                            std::thread::sleep(std::time::Duration::from_secs_f64(
                                1.0 / TARGET_FREQ - elapsed,
                            ));
                        }
                        let extra = if msg.disarm {
                            "D".to_string()
                        } else if msg.arm {
                            "A".to_string()
                        } else if msg.kick {
                            "K".to_string()
                        } else {
                            "".to_string()
                        };

                        let cmd = format!(
                            "p{};Sx{:.2};Sy{:.2};Sz{:.2};Sd{:.0};Kt7000;S.{};\n",
                            msg.id, msg.sx, msg.sy, msg.w, msg.dribble_speed, extra
                        );
                        if !extra.is_empty() {
                            println!("Sending {}", cmd);
                        }

                        if let Err(err) = port.write_all(cmd.as_bytes()) {
                            log::error!("Error writing to serial port: {}", err);
                            sender.send(Err(err.into())).ok();
                        } else if let Err(err) = port.flush() {
                            log::error!("Error flushing serial port: {}", err);
                            sender.send(Err(err.into())).ok();
                        } else {
                            sender.send(Ok(())).ok();
                        }
                        // println!("Sent, dt: {}ms", last_time.elapsed().as_millis());
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
        let (tx, rx) = oneshot::channel();
        self.writer_tx.send((msg, tx))?;
        rx.await?
    }

    pub fn send_no_wait(&mut self, msg: PlayerCmd) {
        let (tx, _) = oneshot::channel();
        self.writer_tx.send((msg, tx)).ok();
    }
}
