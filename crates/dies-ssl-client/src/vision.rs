use anyhow::{Context, Result};
use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;
use tokio::sync::mpsc;

use crate::transport::Transport;

/// Configuration for [`VisionClient`].
pub enum VisionClientConfig {
    /// Receive messages from a TCP socket.
    Tcp { host: String, port: u16 },
    /// Receive messages from a UDP socket.
    Udp { host: String, port: u16 },
    /// In-memory transport for testing.
    InMemory(
        /// Receiver for incoming messages.
        mpsc::UnboundedReceiver<SSL_WrapperPacket>,
    ),
}

/// Async client for SSL Vision.
pub struct VisionClient {
    transport: Transport<SSL_WrapperPacket>,
}

impl VisionClient {
    /// Create a new `SslVisionClient` from the given configuration.
    pub async fn new(config: VisionClientConfig) -> Result<Self> {
        let transport = match config {
            VisionClientConfig::Tcp { host, port } => Transport::tcp(&host, port)
                .await
                .context("Failed to create TCP transport")?,
            VisionClientConfig::Udp { host, port } => Transport::udp(&host, port)
                .await
                .context("Failed to create UDP transport")?,
            VisionClientConfig::InMemory(rx) => Transport::in_memory(rx),
        };
        Ok(Self { transport })
    }

    /// Receive a message from the SSL Vision server.
    pub async fn recv(&mut self) -> Result<SSL_WrapperPacket> {
        self.transport.recv().await
    }
}
