use anyhow::Result;
use dies_core::{EnvEvent, GcRefereeMsg, VisionMsg};
use std::{
    net::{Ipv4Addr, SocketAddr, UdpSocket},
    time::Duration,
};

use dies_protos::Message;

use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use crate::ErSimConfig;

use super::BUF_SIZE;

/// A transport that receives messages from the simulator and the gc running on the
/// local (host) network.
pub struct RecvTransport {
    buf: [u8; BUF_SIZE],
    vision_socket: UdpSocket,
    gc_socket: UdpSocket,
    check_vision_next: bool,
}

impl RecvTransport {
    /// Create a new `RecvTransport`.
    pub fn new(config: &ErSimConfig) -> Result<Self> {
        let ErSimConfig {
            vision_host,
            vision_port,
            gc_host,
            gc_port,
            ..
        } = config;

        let vision_sock = {
            let vision_host = vision_host.parse::<Ipv4Addr>()?;
            let vision_sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
            vision_sock.set_reuse_address(true)?;

            let bind_addr = SockAddr::from(SocketAddr::new(vision_host.into(), *vision_port));
            vision_sock.bind(&bind_addr)?;

            let host = "127.0.0.1".parse::<Ipv4Addr>()?;
            vision_sock.set_multicast_if_v4(&host)?;
            vision_sock.join_multicast_v4(&vision_host, &host)?;

            log::debug!(
                "Bound to simulator vision at {}:{}",
                vision_host,
                vision_port
            );

            vision_sock
        };

        let gc_sock = {
            let gc_sock = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
            gc_sock.set_reuse_address(true)?;

            let bind_addr = SockAddr::from(SocketAddr::new(
                gc_host.parse::<Ipv4Addr>()?.into(),
                *gc_port,
            ));
            gc_sock.bind(&bind_addr)?;

            let host = "127.0.0.1".parse::<Ipv4Addr>()?;
            gc_sock.set_multicast_if_v4(&host)?;
            gc_sock.join_multicast_v4(&gc_host.parse::<Ipv4Addr>()?, &host)?;

            log::debug!("Bound to simulator GC at {}:{}", gc_host, gc_port);

            gc_sock
        };

        let vision_socket = UdpSocket::from(vision_sock);
        vision_socket.set_read_timeout(Some(Duration::from_millis(1)))?;

        let gc_socket = UdpSocket::from(gc_sock);
        gc_socket.set_read_timeout(Some(Duration::from_millis(1)))?;

        Ok(Self {
            buf: [0; BUF_SIZE],
            vision_socket,
            gc_socket,
            check_vision_next: false,
        })
    }

    /// Receive a message from the simulator or the gc.
    ///
    /// This method blocks until a message is received.
    pub fn recv(&mut self) -> Result<EnvEvent> {
        loop {
            self.check_vision_next = !self.check_vision_next;
            if self.check_vision_next {
                if let Ok(len) = self.vision_socket.recv(&mut self.buf) {
                    let msg = VisionMsg::parse_from_bytes(&self.buf[..len])?;
                    return Ok(EnvEvent::VisionMsg(msg));
                }
            } else if let Ok(len) = self.gc_socket.recv(&mut self.buf) {
                let msg = GcRefereeMsg::parse_from_bytes(&self.buf[..len])?;
                return Ok(EnvEvent::GcRefereeMsg(msg));
            }
        }
    }
}
