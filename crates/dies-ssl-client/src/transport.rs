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
    sync::mpsc,
};

enum TransportType<I: Message> {
    Tcp { stream: TcpStream },
    Udp { socket: UdpSocket },
    InMemory { rx: mpsc::UnboundedReceiver<I> },
}

/// A transport for receiving and sending protobuf messages over the network.
///
/// It supports both TCP and UDP.
pub struct Transport<I: Message, O = ()> {
    transport_type: TransportType<I>,
    buf: Vec<u8>,
    incoming_msg_type: PhantomData<I>,
    outgoing_msg_type: PhantomData<O>,
}

impl<I: Message, O> Transport<I, O> {
    pub async fn tcp(host: &str, port: u16) -> Result<Self> {
        let addr = format!("{}:{}", host, port);
        let stream = TcpStream::connect(addr.clone()).await.context(format!(
            "Failed to connect to TCP stream with address {:?}",
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
        let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;
        let raw_socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))
            .context("Failed to create UDP socket")?;
        raw_socket.set_nonblocking(true)?;
        raw_socket.set_reuse_address(true)?;

        let multiaddr = host.parse::<Ipv4Addr>()?;
        let interface = "0.0.0.0".parse::<Ipv4Addr>()?;
        raw_socket
            .join_multicast_v4(&multiaddr, &interface)
            .context("Failed to join multicast group")?;
        raw_socket
            .bind(&addr.into())
            .context(format!("Failed to bind to {}", addr))?;

        let socket = UdpSocket::from_std(raw_socket.into())?;
        Ok(Self {
            transport_type: TransportType::Udp { socket },
            buf: vec![0u8; 2 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        })
    }

    pub fn in_memory(rx: mpsc::UnboundedReceiver<I>) -> Self {
        Self {
            transport_type: TransportType::InMemory { rx },
            buf: vec![0u8; 2 * 1024],
            incoming_msg_type: PhantomData,
            outgoing_msg_type: PhantomData,
        }
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
            TransportType::InMemory { rx, .. } => {
                let msg = rx.recv().await.ok_or_else(|| {
                    anyhow::anyhow!("Failed to receive message from in-memory transport")
                })?;
                Ok(msg)
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
            TransportType::InMemory { .. } => {
                tracing::error!("Sending messages not supported for in-memory transport");
                Ok(())
            }
        }
    }
}
