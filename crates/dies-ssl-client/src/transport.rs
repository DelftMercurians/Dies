use anyhow::{Context, Result};
use socket2::{Domain, Protocol, Socket, Type};
use std::{
    marker::PhantomData,
    net::{Ipv4Addr, SocketAddr},
};

use dies_protos::Message;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpStream, UdpSocket},
};

enum TransportType {
    Tcp { stream: TcpStream },
    Udp { socket: UdpSocket },
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
    pub async fn tcp(host: &str, port: u16) -> Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(addr.clone())
            .await
            .context(format!("Failed to clone addr {:?}", addr))
            .context(format!(
                "Failed to connect to tcp stream with address {:?}",
                addr
            ))?;
        Ok(Self {
            transport_type: TransportType::Tcp { stream },
            buf: vec![0u8; 2 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        })
    }

    pub async fn udp(host: &str, port: u16) -> Result<Self> {
        let addr = format!("{}:{}", host, port)
            .parse::<SocketAddr>()
            .context(format!("Failed to get socketAddr from {}:{}", host, port))?;
        let raw_socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
            .context("Failed to create IPV4, DGRAM, UDP socket")?;
        raw_socket
            .set_nonblocking(true)
            .context("Failed to set socket to nonblocking")?;
        raw_socket
            .set_reuse_address(true)
            .context("Failed to set reuse address to socket")?;

        let multiaddr = host
            .parse::<Ipv4Addr>()
            .context(format!("Failed to get IPV4Addr from {:?}", host))?;
        let interface = "0.0.0.0"
            .parse::<Ipv4Addr>()
            .context("Failed to parse 0.0.0.0 into IPv4Addr")?;
        raw_socket
            .join_multicast_v4(&multiaddr, &interface)
            .context("Failed to join multicast v4 with multiaddr and interface")?;
        raw_socket
            .bind(&addr.into())
            .context("Failed to bind the socket with the address")?;

        let socket = UdpSocket::from_std(raw_socket.into())
            .context("Failed to convert raw socket into UdpSocket")?;
        Ok(Self {
            transport_type: TransportType::Udp { socket },
            buf: vec![0u8; 2 * 1024],
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
                    .context("Failed to read from tcp stream")?;
                let msg = I::parse_from_bytes(&self.buf[..amt])?;
                Ok(msg)
            }
            TransportType::Udp { socket } => {
                let (len, _) = socket
                    .recv_from(&mut self.buf)
                    .await
                    .context("Failed to receive data from upd socket")?;
                let msg = I::parse_from_bytes(&self.buf[..len])?;
                Ok(msg)
            }
        }
    }
}

impl<I: Message, O: Message> Transport<I, O> {
    // TODO: Remove when not needed
    #[allow(dead_code)]
    pub async fn send(&mut self, msg: O) -> Result<()> {
        let buf = msg.write_to_bytes().context("Failed to write to buffer")?;
        match &mut self.transport_type {
            TransportType::Tcp { stream } => {
                stream
                    .write_all(&buf)
                    .await
                    .context("Failed to write to steram")?;
                Ok(())
            }
            TransportType::Udp { socket } => {
                socket
                    .send(&buf)
                    .await
                    .context("Failed to send to upd socket")?;
                Ok(())
            }
        }
    }
}
