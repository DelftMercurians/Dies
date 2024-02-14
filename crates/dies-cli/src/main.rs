use anyhow::Result;
use dies_ssl_client::SslVisionClientConfig;
use dies_world::WorldConfig;
use tokio::{sync::oneshot, time};

use dies_python_rt::PyRuntimeConfig;
use dies_serial_client::list_serial_ports;
use tokio_util::sync::CancellationToken;

mod executor;

use crate::executor::{run, ExecutorConfig};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Warn)
        .init();

    let ports = list_serial_ports()?;
    let port = if !ports.is_empty() {
        if ports.len() == 1 {
            println!("Connecting to serial port {}", ports[0]);
            Some(ports[0].clone())
        } else {
            println!("Available ports:");
            for (idx, port) in ports.iter().enumerate() {
                println!("{}: {}", idx, port);
            }

            // Let user choose port
            loop {
                println!("Enter port number:");
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                let port_idx = input.trim().parse::<usize>()?;
                if port_idx < ports.len() {
                    break Some(ports[port_idx].clone());
                } else {
                    println!("Invalid port number");
                }
            }
        }
    } else {
        println!("No serial ports available, not connecting to basestation");
        None
    };

    let config = ExecutorConfig {
        webui: false,
        robot_ids: std::collections::HashMap::from([(14, 2), (5, 3)]),
        py_config: PyRuntimeConfig {
            workspace: std::env::current_dir().unwrap(),
            package: "dies_test_strat".into(),
            module: "__main__".into(),
            sync: false,
        },
        world_config: WorldConfig {
            is_blue: true,
            initial_opp_goal_x: 1.0,
        },
        vision_config: Some(SslVisionClientConfig {
            host: "localhost".to_string(),
            port: 6078,
            socket_type: dies_ssl_client::SocketType::Tcp,
        }),
        serial_config: match port {
            Some(port) => Some(dies_serial_client::SerialClientConfig {
                port_name: port.clone(),
                ..Default::default()
            }),
            None => None,
        },
    };

    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    run(config, cancel_clone.clone())
        .await
        .expect("Failed to run executor");

    // let ctrc = match tokio::signal::ctrl_c().await {
    //     Ok(fut) => fut,
    //     Err(err) => {
    //         eprintln!("Unable to listen for shutdown signal: {}", err);
    //         // Send stop command
    //         cancel.cancel();
    //         handle.await?;
    //         return Ok(());
    //     }
    // };
    // tokio::select! {
    //     // _ = tokio::signal::ctrl_c() => {}
    //     _ = &mut handle => {}
    // };

    println!("Shutting down");

    // Send stop command
    // cancel.cancel();

    // if !handle.is_finished() {
    //     handle.abort();
    //     handle.await.ok();
    // }

    Ok(())
}
