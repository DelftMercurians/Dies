use std::io::{Read, Write};

use anyhow::{bail, Result};
use interprocess::local_socket::{LocalSocketListener, LocalSocketStream, NameTypeSupport};
use rand::random;

const PACKET_START_MARKER: u8 = 0x02;

/// Local socket IPC -- used to communicate with the python runtime
///
/// Uses domain sockets on linux and named pipes on windows.
///
/// See https://docs.rs/interprocess/1.2.1/interprocess/local_socket
pub struct LocalSocketIpc {
    name: String,
    buf: Vec<u8>,
    listener: LocalSocketListener,
    stream: Option<LocalSocketStream>,
}

impl LocalSocketIpc {
    /// Creates a new IPC connection
    ///
    /// # Errors
    ///
    /// Returns an error if the socket cannot be created.
    pub fn new() -> Result<Self> {
        let (listener, name) = create_listener()?;
        Ok(Self {
            name,
            buf: vec![0; 1024],
            listener,
            stream: None,
        })
    }

    /// Returns the name of the socket -- used to connect to it
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Blocks until a connection is made to our socket
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is already connected or if the connection fails.
    pub fn connect(&mut self) -> Result<()> {
        if self.stream.is_some() {
            bail!("Already connected");
        }
        self.stream = Some(wait_for_conn(&self.listener)?);
        Ok(())
    }

    /// Sends data over the socket.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is not connected.
    pub fn send(&mut self, data: &[u8]) -> Result<()> {
        if self.stream.is_none() {
            bail!("Not connected");
        }
        self.stream.as_mut().unwrap().write_all(data)?;
        Ok(())
    }

    /// Receives data over the socket.
    ///
    /// # Errors
    ///
    /// Returns an error if the socket is not connected.
    pub fn recv(&mut self) -> Result<&[u8]> {
        if self.stream.is_none() {
            bail!("Not connected");
        }
        let mut total_read: usize = 0;
        loop {
            let amt = self
                .stream
                .as_mut()
                .unwrap()
                .read(&mut self.buf[total_read..])?;
            total_read += amt;
            if amt == self.buf.len() {
                log::debug!("Buffer full, resizing");
                self.buf.resize(self.buf.len() * 2, 0);
            } else {
                return Ok(&self.buf[0..total_read]);
            }
        }
    }
}

fn wait_for_conn(listener: &LocalSocketListener) -> Result<LocalSocketStream> {
    for _ in 0..10 {
        match listener.accept() {
            Ok(stream) => return Ok(stream),
            Err(err) => {
                if err.kind() != std::io::ErrorKind::WouldBlock {
                    bail!(err);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    bail!("Failed to accept connection after 10 tries");
}

fn create_listener() -> Result<(LocalSocketListener, String)> {
    for _ in 0..10 {
        let name: String = {
            use NameTypeSupport::*;
            match NameTypeSupport::query() {
                OnlyPaths | Both => format!("/tmp/dies-ipc-{:08x}.sock", random::<u32>()).into(),
                OnlyNamespaced => format!("@dies-ipc-{:08x}.sock", random::<u32>()).into(),
            }
        };
        match LocalSocketListener::bind(name.clone()) {
            Ok(listener) => return Ok((listener, name)),
            Err(err) => {
                if err.kind() != std::io::ErrorKind::AddrInUse {
                    bail!(err);
                }
            }
        }
    }
    bail!("Failed to create local socket listener after 10 tries");
}

fn cobs_encode(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len() + 1);
    let mut zero_index = 0usize;
    output.push(0);

    for (i, &byte) in input.iter().enumerate() {
        if byte == 0 {
            output[zero_index] = (i - zero_index) as u8;
            zero_index = i + 1;
            output.push(0);
        } else {
            output.push(byte);
        }
    }

    output[zero_index] = (output.len() - zero_index) as u8;
    output.push(0);
    output
}

fn cobs_decode(input: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(input.len());
    let mut i = 0;

    while i < input.len() {
        let zero_pos = i + input[i] as usize;
        i += 1;
        while i < zero_pos && i < input.len() {
            output.push(input[i]);
            i += 1;
        }
        if zero_pos < input.len() {
            output.push(0);
        }
    }

    output
}
