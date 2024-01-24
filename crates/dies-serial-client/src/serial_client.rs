use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf};
use tokio_serial::{available_ports, SerialPortBuilderExt, SerialStream};

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
    reader: BufReader<ReadHalf<SerialStream>>,
    writer: WriteHalf<SerialStream>,
}

impl SerialClient {
    /// Create a new `SerialClient`.
    pub fn new(config: SerialClientConfig) -> Result<Self> {
        let mut port = tokio_serial::new(config.port_name, config.baud_rate)
            .timeout(std::time::Duration::from_millis(10))
            .open_native_async()?;

        #[cfg(unix)]
        port.set_exclusive(false)?;

        let (reader, writer) = tokio::io::split(port);
        Ok(Self {
            reader: BufReader::new(reader),
            writer,
        })
    }

    /// Receive a message from the serial port.
    pub async fn recv(&mut self) -> Result<String> {
        let mut buf = String::new();
        self.reader.read_line(&mut buf).await?;
        Ok(buf)
    }

    /// Send a message to the serial port.
    pub async fn send(&mut self, msg: PlayerCmd) -> Result<()> {
        let cmd = format!("Sy{};Sx{};Sz{};S.\n", msg.sy, msg.sx, msg.w);
        self.writer.write_all(cmd.as_bytes()).await?;
        if !cmd.ends_with("\n") {
            self.writer.write_all(b"\n").await?;
        }
        Ok(())
    }
}
