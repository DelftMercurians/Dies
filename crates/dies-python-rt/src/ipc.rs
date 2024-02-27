use anyhow::Result;
use std::{collections::HashSet, net::SocketAddr, time::Duration};
use tokio::net::UdpSocket;

use crate::{RuntimeEvent, RuntimeMsg};

/// An IPC connection to a child process.
pub struct IpcSocket {
    socket: UdpSocket,
    host: String,
    port: u16,
    peers: HashSet<SocketAddr>,
    read_buf: Vec<u8>,
}

impl IpcSocket {
    /// Create a new IPC socket to which a child process can connect.
    ///
    /// # Errors
    ///
    /// Returns an error if the connection cannot be created.
    pub async fn new() -> Result<IpcSocket> {
        let host = "127.0.0.1".to_string();
        let socket = UdpSocket::bind(format!("{host}:0")).await?;
        let port = socket.local_addr()?.port();
        Ok(IpcSocket {
            socket,
            host,
            port,
            read_buf: vec![0; 2 * 1024],
            peers: HashSet::new(),
        })
    }

    /// Get the port
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Get the host
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Wait for a message from the child process with a timeout.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be received or the timeout expires.
    pub async fn wait(&mut self, timeout: Duration) -> Result<()> {
        let mut buf = [0; 2 * 1024];
        let (_, peer) = tokio::time::timeout(timeout, self.socket.recv_from(&mut buf)).await??;
        self.peers.insert(peer);
        Ok(())
    }

    /// Send a message to the child process.
    ///
    /// # Errors
    ///
    /// Returns an error if the message cannot be sent.
    pub async fn send(&mut self, data: &RuntimeMsg) -> Result<()> {
        let data = serde_json::to_string(data)?;
        let bytes = data.as_bytes();
        for peer in &self.peers {
            self.socket.send_to(bytes, peer).await?;
        }
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
    pub async fn recv(&mut self) -> Result<RuntimeEvent> {
        let (n, peer) = self.socket.recv_from(&mut self.read_buf).await?;
        self.peers.insert(peer);
        serde_json::from_slice(&self.read_buf[..n]).map_err(Into::into)
    }
}
