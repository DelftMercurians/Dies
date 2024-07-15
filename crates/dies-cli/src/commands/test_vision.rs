use std::{
    net::{SocketAddr},
};

use anyhow::Result;
use dies_ssl_client::VisionClientConfig;
use network_interface::{NetworkInterface, NetworkInterfaceConfig};
use tokio::time::Instant;

use crate::cli::VisionType;

pub async fn test_vision(
    vision: VisionType,
    vision_addr: SocketAddr,
    interface: Option<String>,
) -> Result<()> {
    env_logger::init();

    let network_interfaces = NetworkInterface::show().unwrap();
    println!("Network interfaces:");
    for itf in network_interfaces.iter() {
        println!(" - {}: {:?}", itf.name, itf.addr);
    }

    let conf = match vision {
        VisionType::None => None,
        VisionType::Tcp => Some(VisionClientConfig::Tcp {
            host: vision_addr.ip().to_string(),
            port: vision_addr.port(),
        }),
        VisionType::Udp => Some(VisionClientConfig::Udp {
            host: vision_addr.ip().to_string(),
            port: vision_addr.port(),
            interface,
        }),
    }
    .ok_or(anyhow::anyhow!("Invalid vision configuration"))?;

    let mut ssl_client = dies_ssl_client::VisionClient::new(conf).await?;
    println!("Starting vision client");

    let start = Instant::now();
    loop {
        let packet = ssl_client.recv().await?;
        println!("Received packet: {:?}", packet);
        if start.elapsed().as_secs() > 5 {
            break;
        }
    }

    Ok(())
}
