use std::net::SocketAddr;

use anyhow::Result;
use dies_ssl_client::{ConnectionConfig, SslClientConfig};
use network_interface::{NetworkInterface, NetworkInterfaceConfig};

use crate::cli::ConnectionMode;

pub async fn test_vision(
    mode: ConnectionMode,
    vision_addr: SocketAddr,
    gc_addr: SocketAddr,
    interface: Option<String>,
) -> Result<()> {
    env_logger::init();

    let network_interfaces = NetworkInterface::show().unwrap();
    println!("Network interfaces:");
    for itf in network_interfaces.iter() {
        println!(" - {}: {:?}", itf.name, itf.addr);
    }

    let vision_conf = match mode {
        ConnectionMode::None => None,
        ConnectionMode::Tcp => Some(ConnectionConfig::Tcp {
            host: vision_addr.ip().to_string(),
            port: vision_addr.port(),
        }),
        ConnectionMode::Udp => Some(ConnectionConfig::Udp {
            host: vision_addr.ip().to_string(),
            port: vision_addr.port(),
            interface: interface.clone(),
        }),
    }
    .ok_or(anyhow::anyhow!("Invalid vision configuration"))?;

    let gc_conf = match mode {
        ConnectionMode::None => None,
        ConnectionMode::Tcp => Some(ConnectionConfig::Tcp {
            host: gc_addr.ip().to_string(),
            port: gc_addr.port(),
        }),
        ConnectionMode::Udp => Some(ConnectionConfig::Udp {
            host: gc_addr.ip().to_string(),
            port: gc_addr.port(),
            interface,
        }),
    }
    .ok_or(anyhow::anyhow!("Invalid gc configuration"))?;

    let mut ssl_client = dies_ssl_client::VisionClient::new(SslClientConfig {
        vision: vision_conf,
        gc: gc_conf,
    })
    .await?;
    println!("Starting vision client");

    let mut received_gc = false;
    let mut received_vision = false;
    while !received_gc || !received_vision {
        let packet = ssl_client.recv().await?;
        match packet {
            dies_ssl_client::SslMessage::Vision(_) => {
                if !received_vision {
                    println!("Received vision packet");
                    received_vision = true;
                }
            }
            dies_ssl_client::SslMessage::Referee(_) => {
                if !received_gc {
                    println!("Received GC packet");
                    received_gc = true;
                }
            }
            dies_ssl_client::SslMessage::Tracker(_) => {
                println!("Received tracker packet");
            }
        }
    }

    Ok(())
}
