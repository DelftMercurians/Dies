use std::{
    marker::PhantomData,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    time::Duration,
};

use anyhow::{Context, Result};
use dies_protos::Message;
use network_interface::{NetworkInterface, NetworkInterfaceConfig};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpStream, UdpSocket},
};

enum TransportType {
    Tcp {
        stream: TcpStream,
    },
    Udp {
        socket: UdpSocket,
        logged_first_packet: bool,
    },
    Mock,
}

/// A transport for receiving and sending protobuf messages over the network.
///
/// It supports both TCP and UDP.
pub struct Transport<I: Message, O = ()> {
    transport_type: TransportType,
    buf: Vec<u8>,
    incoming_msg_type: PhantomData<I>,
    outgoing_msg_type: PhantomData<O>,
}

impl<I: Message, O> Transport<I, O> {
    pub fn mock() -> Self {
        Self {
            transport_type: TransportType::Mock,
            buf: vec![0u8; 64 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        }
    }

    pub async fn tcp(host: &str, port: u16) -> Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(addr.clone()).await.context(format!(
            "Failed to connect to TCP stream with address {:?}",
            addr
        ))?;
        Ok(Self {
            transport_type: TransportType::Tcp { stream },
            buf: vec![0u8; 32 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        })
    }

    pub async fn udp(host: &str, port: u16, interface: Option<String>) -> Result<Self> {
        let raw_socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
            .context("Failed to create UDP socket")?;
        raw_socket.set_nonblocking(true)?;
        raw_socket.set_reuse_address(true)?;

        // Bind to INADDR_ANY so the socket also receives broadcast/unicast
        // packets on this port (fields sometimes fall back to broadcast).
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), port);
        raw_socket
            .bind(&addr.into())
            .context(format!("Failed to bind to {}", addr))?;

        let dest = host.parse::<Ipv4Addr>()?;
        if dest.is_multicast() {
            Self::join_multicast(&raw_socket, dest, interface)?;
        }

        let socket = UdpSocket::from_std(raw_socket.into())?;
        Ok(Self {
            transport_type: TransportType::Udp {
                socket,
                logged_first_packet: false,
            },
            buf: vec![0u8; 32 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        })
    }

    /// Joins `group` on the named interface (or the first eth*/en* one if
    /// `None`). The special name "all" joins best-effort on every IPv4
    /// interface instead.
    fn join_multicast(socket: &Socket, group: Ipv4Addr, interface: Option<String>) -> Result<()> {
        let network_interfaces = NetworkInterface::show()?;
        let first_ipv4 = |itf: &NetworkInterface| {
            itf.addr.iter().find_map(|addr| match addr.ip() {
                IpAddr::V4(v4) => Some(v4),
                IpAddr::V6(_) => None,
            })
        };

        if interface.as_deref() == Some("all") {
            let mut joined = 0;
            for itf in &network_interfaces {
                let Some(if_ip) = first_ipv4(itf) else {
                    continue;
                };
                if socket.join_multicast_v4(&group, &if_ip).is_ok() {
                    log::debug!(
                        "Joined multicast group {} on {} ({})",
                        group,
                        itf.name,
                        if_ip
                    );
                    joined += 1;
                }
            }
            if joined == 0 {
                anyhow::bail!("Failed to join multicast group {} on any interface", group);
            }
            return Ok(());
        }

        let itf = network_interfaces
            .iter()
            .find(|itf| {
                if let Some(if_name) = interface.as_ref() {
                    itf.name == *if_name
                } else {
                    (itf.name.starts_with("eth") || itf.name.starts_with("en"))
                        && first_ipv4(itf).is_some()
                }
            })
            .ok_or(anyhow::anyhow!("No suitable network interface found"))?;
        let if_ip = first_ipv4(itf).ok_or(anyhow::anyhow!(
            "No IPv4 address found for network interface {}",
            itf.name
        ))?;
        socket
            .join_multicast_v4(&group, &if_ip)
            .context("Failed to join multicast group")?;
        log::debug!(
            "Joined multicast group {} on {} ({})",
            group,
            itf.name,
            if_ip
        );
        Ok(())
    }

    pub async fn recv(&mut self) -> Result<I> {
        match &mut self.transport_type {
            TransportType::Tcp { stream } => {
                let amt = stream
                    .read(&mut self.buf)
                    .await
                    .context("Failed to read from TCP stream")?;
                let msg = I::parse_from_bytes(&self.buf[..amt])?;
                Ok(msg)
            }
            TransportType::Udp {
                socket,
                logged_first_packet,
            } => {
                let (len, src) = socket
                    .recv_from(&mut self.buf)
                    .await
                    .context("Failed to receive data from UDP socket")?;
                if !*logged_first_packet {
                    *logged_first_packet = true;
                    let port = socket.local_addr().map(|a| a.port()).unwrap_or(0);
                    log::info!("First packet on UDP port {} from {}", port, src);
                }
                let msg = I::parse_from_bytes(&self.buf[..len])?;
                Ok(msg)
            }
            TransportType::Mock => {
                tokio::time::sleep(Duration::from_millis(10)).await;
                Err(anyhow::anyhow!("Mock transport does not support recv"))
            }
        }
    }
}

impl<I: Message, O: Message> Transport<I, O> {
    // TODO: Remove when not needed
    #[allow(dead_code)]
    pub async fn send(&mut self, msg: O) -> Result<()> {
        let buf = msg.write_to_bytes()?;
        match &mut self.transport_type {
            TransportType::Tcp { stream } => {
                stream
                    .write_all(&buf)
                    .await
                    .context("Failed to write to TCP stream")?;
                Ok(())
            }
            TransportType::Udp { socket, .. } => {
                socket
                    .send(&buf)
                    .await
                    .context("Failed to send on UDP socket")?;
                Ok(())
            }
            TransportType::Mock => Err(anyhow::anyhow!("Mock transport does not support send")),
        }
    }
}
