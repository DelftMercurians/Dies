use anyhow::Result;
use dies_protos::Message;
use std::net::UdpSocket;

use crate::ErSimConfig;

use super::BUF_SIZE;

pub struct RecvTransport {
    buf: [u8; BUF_SIZE],
    rx_socket: UdpSocket,
}

impl RecvTransport {
    pub fn new(config: &ErSimConfig) -> Result<Self> {
        let rx_socket = UdpSocket::bind(format!("127.0.0.1:{}", config.bridge_port))?;
        log::debug!("Bound bridge rx socket to {:?}", rx_socket.local_addr());

        Ok(Self {
            buf: [0; BUF_SIZE],
            rx_socket,
        })
    }

    pub fn recv(&mut self) -> Vec<dies_core::EnvEvent> {
        let data = match self.rx_socket.recv(&mut self.buf) {
            Ok(data) => data,
            Err(err) => {
                if err.kind() == std::io::ErrorKind::WouldBlock {
                    return vec![];
                } else {
                    log::error!("Failed to receive data: {}", err);
                    return vec![];
                }
            }
        };

        // Check first byte for packet type
        match self.buf[0] {
            0 => {
                // Vision message
                let msg = match dies_protos::ssl_vision_wrapper::SSL_WrapperPacket::parse_from_bytes(
                    &self.buf[1..data],
                ) {
                    Ok(msg) => msg,
                    Err(err) => {
                        log::error!("Failed to parse vision message: {}", err);
                        return vec![];
                    }
                };
                vec![dies_core::EnvEvent::VisionMsg(msg)]
            }
            1 => {
                // GC message
                let msg = match dies_protos::ssl_gc_referee_message::Referee::parse_from_bytes(
                    &self.buf[1..data],
                ) {
                    Ok(msg) => msg,
                    Err(err) => {
                        log::error!("Failed to parse GC message: {}", err);
                        return vec![];
                    }
                };
                vec![dies_core::EnvEvent::GcRefereeMsg(msg)]
            }
            _ => {
                log::error!("Received message with unknown type: {}", self.buf[0]);
                vec![]
            }
        }
    }
}
