use anyhow::Result;
use dies_protos::ssl_vision_wrapper::SSL_WrapperPacket;

use crate::transport::Transport;

pub enum SocketType {
    Tcp,
    Udp,
}

pub struct SslVisionClientConfig {
    pub host: String,
    pub port: u16,
    pub socket_type: SocketType,
}

/// Async client for the SSL Vision server.
///
/// # Example
///
/// ```no_run
/// use dies_ssl_client::{SslVisionClient, SslVisionClientConfig, SocketType};
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let config = SslVisionClientConfig {
///         host: "localhost".to_string(),
///         port: 10020,
///         socket_type: SocketType::Udp,
///     };
///     let mut client = SslVisionClient::new(config).await?;
///     let msg = client.recv().await?;
///     println!("{:?}", msg);
/// }
/// ```
pub struct SslVisionClient {
    transport: Transport<SSL_WrapperPacket>,
}

impl SslVisionClient {
    /// Create a new `SslVisionClient`.
    pub async fn new(config: SslVisionClientConfig) -> Result<Self> {
        let transport = match config.socket_type {
            SocketType::Tcp => Transport::tcp(&config.host, config.port).await?,
            SocketType::Udp => Transport::udp(&config.host, config.port).await?,
        };
        Ok(Self { transport })
    }

    /// Receive a message from the SSL Vision server.
    pub async fn recv(&mut self) -> Result<SSL_WrapperPacket> {
        self.transport.recv().await
    }
}
