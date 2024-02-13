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
            loop {
                match rx.blocking_recv() {
                    Some((msg, sender)) => {
                        let cmd = format!(
                            "p{};Sx{};Sy{};Sz{};Sd0;S.;\n",
                            msg.id, msg.sx, msg.sy, msg.w
                        );
                        if let Err(err) = port.write_all(cmd.as_bytes()) {
                            sender.send(Err(err.into())).unwrap();
                        } else if let Err(err) = port.flush() {
                            sender.send(Err(err.into())).unwrap();
                        } else {
                            sender.send(Ok(())).unwrap();
                        }
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
}
