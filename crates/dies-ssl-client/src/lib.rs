mod transport;

use anyhow::{Context, Result};
use dies_protos::{
    ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket,
    ssl_vision_wrapper_tracked::TrackerWrapperPacket,
};

use crate::transport::Transport;

/// Configuration for [`VisionClient`].
#[derive(Debug, Clone)]

pub enum ConnectionConfig {
    /// Receive messages from a TCP socket.
    Tcp {
        host: String,
        port: u16,
    },
    /// Receive messages from a UDP socket.
    Udp {
        host: String,
        port: u16,
        interface: Option<String>,
    },
    Mock,
}

#[derive(Debug, Clone)]
pub struct SslClientConfig {
    pub vision: ConnectionConfig,
    pub gc: ConnectionConfig,
}

/// Async client for SSL Vision.
pub struct VisionClient {
    vision_transport: Transport<SSL_WrapperPacket>,
    gc_transport: Transport<Referee>,
    tracker_transport: Option<Transport<TrackerWrapperPacket>>,
}

#[derive(Debug)]
pub enum SslMessage {
    Vision(SSL_WrapperPacket),
    Referee(Referee),
    Tracker(TrackerWrapperPacket),
}

impl VisionClient {
    /// Create a new `SslVisionClient` from the given configuration.
    pub async fn new(config: SslClientConfig) -> Result<Self> {
        let vision_transport = match config.vision.clone() {
            ConnectionConfig::Tcp { host, port } => Transport::tcp(&host, port)
                .await
                .context("Failed to create TCP transport")?,
            ConnectionConfig::Udp {
                host,
                port,
                interface,
            } => Transport::udp(&host, port, interface)
                .await
                .context("Failed to create UDP transport for vision")?,
            ConnectionConfig::Mock => Transport::mock(),
        };

        let gc_transport = match config.gc {
            ConnectionConfig::Tcp { host, port } => Transport::tcp(&host, port)
                .await
                .context("Failed to create TCP transport")?,
            ConnectionConfig::Udp {
                host,
                port,
                interface,
            } => Transport::udp(&host, port, interface)
                .await
                .context("Failed to create UDP transport for GC")?,
            ConnectionConfig::Mock => Transport::mock(),
        };

        let tracker_transport = match config.vision {
            ConnectionConfig::Udp { interface, .. } => {
                Transport::udp("224.5.23.2", 10010, interface).await.ok()
            }
            _ => None,
        };

        Ok(Self {
            vision_transport,
            gc_transport,
            tracker_transport,
        })
    }

    /// Receive a message from either the vision or the referee.
    pub async fn recv(&mut self) -> Result<SslMessage> {
        if let Some(tracker_transport) = &mut self.tracker_transport {
            tokio::select! {
                msg = self.vision_transport.recv() => Ok(SslMessage::Vision(msg?)),
                msg = self.gc_transport.recv() => Ok(SslMessage::Referee(msg?)),
                msg = tracker_transport.recv() => Ok(SslMessage::Tracker(msg?)),
            }
        } else {
            tokio::select! {
                msg = self.vision_transport.recv() => Ok(SslMessage::Vision(msg?)),
                msg = self.gc_transport.recv() => Ok(SslMessage::Referee(msg?)),
            }
        }
    }
}
