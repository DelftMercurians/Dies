use std::{net::{Ipv4Addr, SocketAddr, UdpSocket}, str::FromStr};

use anyhow::Result;
use dies_ssl_client::VisionClientConfig;
use network_interface::{NetworkInterface, NetworkInterfaceConfig};
use socket2::{Domain, Protocol, Socket, Type};
use tokio::time::Instant;

use crate::cli::VisionType;

pub async fn test_vision(vision: VisionType, vision_addr: SocketAddr) -> Result<()> {
    let network_interfaces = NetworkInterface::show().unwrap();

    for itf in network_interfaces.iter() {
        println!("{:?}", itf);
    }

    
    let host = "224.5.23.2";
    let port = 10006;
    let addr = format!("{}:{}", host, port).parse::<SocketAddr>()?;

    let socket = UdpSocket::bind(addr).expect("Couldn't bind proposer UDP socket");
    socket
        .join_multicast_v4(
            &Ipv4Addr::from_str("224.5.23.2").unwrap(),
            &Ipv4Addr::UNSPECIFIED,
        )
        .expect("Could not join multicast group A");

    loop {
        let mut buf = vec![0u8; 2 * 1024];
        let (len, _) = socket.recv_from(&mut buf)?;
        println!("Received {} bytes", len);
        if len > 0 {
            break;
        }
    }

    // env_logger::init();

    // let conf = match vision {
    //     VisionType::None => None,
    //     VisionType::Tcp => Some(VisionClientConfig::Tcp {
    //         host: vision_addr.ip().to_string(),
    //         port: vision_addr.port(),
    //     }),
    //     VisionType::Udp => Some(VisionClientConfig::Udp {
    //         host: vision_addr.ip().to_string(),
    //         port: vision_addr.port(),
    //     }),
    // }
    // .ok_or(anyhow::anyhow!("Invalid vision configuration"))?;

    // let mut ssl_client = dies_ssl_client::VisionClient::new(conf).await?;
    // println!("Starting vision client");

    // let start = Instant::now();
    // loop {
    //     let packet = ssl_client.recv().await?;
    //     println!("Received packet: {:?}", packet);
    //     if start.elapsed().as_secs() > 5 {
    //         break;
    //     }
    // }

    Ok(())
}
