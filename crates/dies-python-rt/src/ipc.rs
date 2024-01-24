use anyhow::{bail, Result};
use std::time::Duration;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, ReadHalf, WriteHalf},
    net::{TcpListener, TcpStream},
};

/// A listener that opens a TCP socket and waits for a connection from a child process.
pub struct IpcListener {
    host: String,
    port: u16,
    listener: TcpListener,
}

/// An IPC connection to a child process.
pub struct IpcConnection {
    reader: BufReader<ReadHalf<TcpStream>>,
    writer: WriteHalf<TcpStream>,
}

impl IpcListener {
    /// Create a new `IpcCodec`.
    ///
    /// Creates a new TCP listener on a random port on the local host.
    ///
    /// # Errors
    ///
    /// Returns an error if the listener cannot be created.
    pub async fn new() -> Result<IpcListener> {
        let listener = TcpListener::bind("0.0.0.0:0").await?;
        let host = listener.local_addr()?.ip().to_string();
        let port = listener.local_addr()?.port();
        Ok(IpcListener {
            host,
            port,
            listener,
        })
    }

    /// Get the hostname where the codec is listening.
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get the port where the codec is listening.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Wait for a connection from a child process or timeout.
    ///
    /// If successful, returns a pair of [`IpcSender`] and [`IpcReceiver`] that can be
    /// used to communicate with the child process.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be accepted or the timeout expires.
    pub async fn wait_for_conn(self, timeout: Duration) -> Result<IpcConnection> {
        tokio::select! {
            res = self.listener.accept() => {
                let (stream, _) = res?;
                let (reader, writer) = tokio::io::split(stream);
                let reader = BufReader::new(reader);
                Ok(IpcConnection { reader, writer })
            }
            _ = tokio::time::sleep(timeout) => {
                bail!("Timeout waiting for connection from child process");
            }
        }
    }
}

impl IpcConnection {
    /// Send a message to the child process.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be sent.
    pub async fn send(&mut self, data: &str) -> Result<()> {
        self.writer.write_all(&data.as_bytes()).await?;
        if !data.ends_with("\n") {
            self.writer.write_all(b"\n").await?;
        }
        self.writer.flush().await?;
        Ok(())
    }

    /// Receive a message from the child process.
    ///
    /// This method blocks until a message is received.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be
    /// received.
    pub async fn recv(&mut self) -> Result<String> {
        let mut line = String::new();
        self.reader.read_line(&mut line).await?;
        Ok(line)
    }
}
