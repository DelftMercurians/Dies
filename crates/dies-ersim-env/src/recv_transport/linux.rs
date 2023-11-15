use anyhow::Result;
use polling::{Events, Poller};
use std::net::{Ipv4Addr, SocketAddr, UdpSocket};

use dies_protos::Message;

use socket2::{Domain, Protocol, SockAddr, Socket, Type};

use crate::ErSimConfig;

use super::BUF_SIZE;

const VISION_SOCK_KEY: usize = 1;
const GC_SOCK_KEY: usize = 2;

pub struct RecvTransport {
    buf: [u8; BUF_SIZE],
    vision_socket: UdpSocket,
    gc_socket: UdpSocket,
    poller: Poller,
    events: Events,
}

impl RecvTransport {
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
        vision_socket.set_nonblocking(true)?;

        let gc_socket = UdpSocket::from(gc_sock);
        gc_socket.set_nonblocking(true)?;

        let poller = Poller::new()?;
        // Safety: We just created these sockets, so they are valid. We have to make
        //         sure to remove them from the poller before they are dropped.
        //         See the Drop impl for ErSimEnv.
        unsafe {
            poller.add(&vision_socket, polling::Event::readable(VISION_SOCK_KEY))?;
            poller.add(&gc_socket, polling::Event::readable(GC_SOCK_KEY))?;
        }

        Ok(Self {
            buf: [0; BUF_SIZE],
            vision_socket,
            gc_socket,
            poller,
            events: Events::new(),
        })
    }

    pub fn recv(&mut self) -> Vec<dies_core::EnvEvent> {
        if self.events.is_empty() {
            match self.poller.wait(&mut self.events, None) {
                Ok(_) => {}
                Err(err) => {
                    log::error!("Failed to poll sockets: {}", err);
                    return vec![];
                }
            }
        }

        let result: Vec<_> = self
            .events
            .iter()
            .filter_map(|ev| {
                if ev.key == VISION_SOCK_KEY {
                    let amt = match self.vision_socket.recv(&mut self.buf) {
                        Ok(amt) => amt,
                        Err(err) => {
                            if err.kind() == std::io::ErrorKind::WouldBlock {
                                return None;
                            } else {
                                log::error!("Failed to receive vision message: {}", err);
                                return None;
                            }
                        }
                    };

                    match dies_core::VisionMsg::parse_from_bytes(&self.buf[..amt]) {
                        Ok(msg) => return Some(dies_core::EnvEvent::VisionMsg(msg)),
                        Err(err) => {
                            log::error!("Failed to parse vision message: {}", err);
                            return None;
                        }
                    };
                }
                if ev.key == GC_SOCK_KEY {
                    let amt = match self.gc_socket.recv(&mut self.buf) {
                        Ok(amt) => amt,
                        Err(err) => {
                            if err.kind() == std::io::ErrorKind::WouldBlock {
                                return None;
                            } else {
                                log::error!("Failed to receive GC message: {}", err);
                                return None;
                            }
                        }
                    };

                    match dies_core::GcRefereeMsg::parse_from_bytes(&self.buf[..amt]) {
                        Ok(msg) => return Some(dies_core::EnvEvent::GcRefereeMsg(msg)),
                        Err(err) => {
                            log::error!("Failed to parse GC message: {}", err);
                            return None;
                        }
                    };
                }
                None
            })
            .collect();

        self.events.clear();

        return result;
    }
}
