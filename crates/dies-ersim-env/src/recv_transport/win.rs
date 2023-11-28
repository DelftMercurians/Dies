use anyhow::{anyhow, Result};
use dies_core::EnvEvent;
use dies_protos::Message;
use std::net::UdpSocket;

use crate::ErSimConfig;

use super::BUF_SIZE;

/// A UDP socket that receives messages from the bridge.
pub struct RecvTransport {
    buf: [u8; BUF_SIZE],
    rx_socket: UdpSocket,
}

impl RecvTransport {
    /// Create a new `RecvTransport`.
    pub fn new(config: &ErSimConfig) -> Result<Self> {
        let rx_socket = UdpSocket::bind(format!("127.0.0.1:{}", config.bridge_port))?;
        log::debug!("Bound bridge rx socket to {:?}", rx_socket.local_addr());

        Ok(Self {
            buf: [0; BUF_SIZE],
            rx_socket,
        })
    }

    /// Receive a message from the bridge.
    ///
    /// This method blocks until a message is received.
    pub fn recv(&mut self) -> Result<EnvEvent> {
        let data = self.rx_socket.recv(&mut self.buf)?;

        // Check first byte for packet type
        match self.buf[0] {
            0 => {
                // Vision message
                let msg = dies_protos::ssl_vision_wrapper::SSL_WrapperPacket::parse_from_bytes(
                    &self.buf[1..data],
                )?;
                Ok(EnvEvent::VisionMsg(msg))
            }
            1 => {
                // GC message
                let msg = dies_protos::ssl_gc_referee_message::Referee::parse_from_bytes(
                    &self.buf[1..data],
                )?;
                Ok(EnvEvent::GcRefereeMsg(msg))
            }
            _ => Err(anyhow!(
                "Received message with unknown type: {}",
                self.buf[0]
            )),
        }
    }
}
