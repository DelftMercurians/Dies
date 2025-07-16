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
    Tcp { stream: TcpStream },
    Udp { socket: UdpSocket },
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
        let network_interfaces = NetworkInterface::show()?;
        // Select eth* or en* interface
        let interface = network_interfaces
            .iter()
            .find(|itf| {
                if let Some(if_name) = interface.as_ref() {
                    itf.name == *if_name
                } else {
                    itf.name.starts_with("eth") || itf.name.starts_with("en")
                }
            })
            .ok_or(anyhow::anyhow!("No suitable network interface found"))?;
        let if_ip = interface
            .addr
            .iter()
            .map(|addr| addr.ip())
            .find_map(|addr| match addr {
                IpAddr::V4(v4) => Some(v4),
                IpAddr::V6(_) => None,
            })
            .ok_or(anyhow::anyhow!(
                "No IPv4 address found for network interface {}",
                interface.name
            ))?;

        let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;
        let raw_socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
            .context("Failed to create UDP socket")?;
        raw_socket.set_nonblocking(true)?;
        raw_socket.set_reuse_address(true)?;

        let multiaddr = host.parse::<Ipv4Addr>()?;
        raw_socket
            .join_multicast_v4(&multiaddr, &if_ip)
            .context("Failed to join multicast group")?;
        raw_socket
            .bind(&addr.into())
            .context(format!("Failed to bind to {}", addr))?;

        let socket = UdpSocket::from_std(raw_socket.into())?;
        Ok(Self {
            transport_type: TransportType::Udp { socket },
            buf: vec![0u8; 32 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        })
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
            TransportType::Udp { socket } => {
                let (len, _) = socket
                    .recv_from(&mut self.buf)
                    .await
                    .context("Failed to receive data from UDP socket")?;
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
            TransportType::Udp { socket } => {
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
