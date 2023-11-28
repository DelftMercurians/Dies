use anyhow::{bail, Result};
use std::{
    io::{BufRead, BufReader, ErrorKind, LineWriter, Write},
    net::{TcpListener, TcpStream},
    time::{Duration, Instant},
};

/// A listener that opens a TCP socket and waits for a connection from a child process.
pub struct IpcListener {
    host: String,
    port: u16,
    listener: TcpListener,
}

/// A sender that sends messages to a child process.
pub struct IpcSender {
    writer: LineWriter<TcpStream>,
}

/// A receiver that receives messages from a child process.
pub struct IpcReceiver {
    reader: BufReader<TcpStream>,
}

impl IpcListener {
    /// Create a new `IpcCodec`.
    ///
    /// Creates a new TCP listener on a random port on the local host.
    ///
    /// # Errors
    ///
    /// Returns an error if the listener cannot be created.
    pub fn new() -> Result<IpcListener> {
        let listener = TcpListener::bind("0.0.0.0:0")?;
        listener.set_nonblocking(true)?;
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
    pub fn wait_for_conn(self, timeout: Duration) -> Result<(IpcSender, IpcReceiver)> {
        let start = Instant::now();
        loop {
            match self.listener.accept() {
                Ok((stream, _)) => {
                    let reader = BufReader::new(stream.try_clone()?);
                    let writer = LineWriter::new(stream);
                    break Ok((IpcSender { writer }, IpcReceiver { reader }));
                }
                Err(err) => {
                    if err.kind() != ErrorKind::WouldBlock {
                        bail!("Failed to accept connection: {}", err);
                    }
                    if start.elapsed() > timeout {
                        bail!("Timeout waiting for connection");
                    }
                }
            }
        }
    }
}

impl IpcSender {
    /// Send a message to the child process.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be sent.
    pub fn send(&mut self, data: &str) -> Result<()> {
        self.writer.write(&data.as_bytes())?;
        self.writer.write(&[b'\n'])?;
        Ok(())
    }
}

impl IpcReceiver {
    /// Receive a message from the child process.
    ///
    /// This method blocks until a message is received.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be
    /// received.
    pub fn recv(&mut self) -> Result<String> {
        let mut line = String::new();
        self.reader.read_line(&mut line)?;
        Ok(line)
    }
}
