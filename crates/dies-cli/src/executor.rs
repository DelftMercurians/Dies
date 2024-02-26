use std::{
    collections::{HashMap, HashSet},
    pin::pin,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use futures::{future, FutureExt};

use dies_core::PlayerCmd;
use dies_python_rt::{PyRuntime, PyRuntimeConfig, RuntimeEvent, RuntimeMsg};
use dies_serial_client::{SerialClient, SerialClientConfig};
use dies_ssl_client::{SslVisionClient, SslVisionClientConfig};
use dies_webui::spawn_webui;
use dies_world::{WorldConfig, WorldTracker};

pub struct ExecutorConfig {
    pub py_config: PyRuntimeConfig,
    pub world_config: WorldConfig,
    pub vision_config: Option<SslVisionClientConfig>,
    pub serial_config: Option<SerialClientConfig>,
    pub webui: bool,
    /// Maps vision IDs to robot IDs
    pub robot_ids: HashMap<u32, u32>,
}

pub async fn run(config: ExecutorConfig) -> Result<()> {
    let mut tracker = WorldTracker::new(config.world_config);
    let mut runtime = PyRuntime::new(config.py_config).await?;
    let mut vision = match config.vision_config {
        Some(vision_config) => Some(SslVisionClient::new(vision_config).await?),
        None => None,
    };
    let mut serial = match config.serial_config {
        Some(serial_config) => Some(SerialClient::new(serial_config)?),
        None => None,
    };
    let robot_ids = config.robot_ids;

    // Launch webui
    let (webui_sender, webui_handle) = if config.webui {
        let (webui_sender, webui_handle) = spawn_webui();
        (Some(webui_sender), Some(webui_handle))
    } else {
        (None, None)
    };

    let mut ctrlc = pin!(tokio::signal::ctrl_c());

    let mut fail: HashMap<u32, bool> = HashMap::new();
    let mut robots: HashSet<u32> = HashSet::new();
    loop {
        let vision_msg_fut = if let Some(vision) = &mut vision {
            vision.recv().map(Some).boxed()
        } else {
            future::ready(None).boxed()
        };

        tokio::select! {
            _ = &mut ctrlc => {
                println!("Received Ctrl-C");
                break;
            }
            vision_msg = vision_msg_fut => {
                match vision_msg {
                    Some(Ok(vision_msg)) => {
                        tracker.update_from_vision(&vision_msg);
                        if let Some(world_data) = tracker.get() {
                            // Failsafe: if one of our robots is not detected, we send stop to runtime
                            for player in world_data.own_players.iter() {
                                if  SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as f64 - player.timestamp > 0.5 {
                                    if fail.get(&player.id) == Some(&false) {
                                        fail.insert(player.id, true);
                                        if let Some(serial) = &mut serial {
                                            tracing::warn!("Failsafe: sending stop to runtime");
                                            serial.send_no_wait(PlayerCmd::zero(*robot_ids.get(&player.id).unwrap_or(&0)));
                                        }
                                    }
                                } else {
                                    fail.insert(player.id, false);
                                }
                            }

                            // Send update to runtime
                            if let Err(err) = runtime.send(&dies_python_rt::RuntimeMsg::World(world_data.clone())).await {
                                tracing::error!("Failed to send world data to runtime: {}", err);
                            }

                            // Send update to webui
                            if let Some(ref webui_sender) = webui_sender {
                                if let Err(err) = webui_sender.send(world_data) {
                                    tracing::error!("Failed to send world data to webui: {}", err);
                                }
                            }
                        }
                    }
                    Some(Err(err)) => {
                        tracing::error!("Failed to receive vision msg: {}", err);
                    }
                    _ => {}
                }
            }
            runtime_msg = runtime.recv() => {
                match runtime_msg {
                    Ok(RuntimeEvent::PlayerCmd(mut cmd)) => {
                        if let Some(serial) = &mut serial {
                            let rid = *robot_ids.get(&cmd.id).unwrap_or(&0);
                            robots.insert(rid);
                            cmd.id = rid;
                            if fail.get(&cmd.id) == Some(&true) {
                                tracing::error!("Failsafe: not sending player cmd");
                                serial.send_no_wait(PlayerCmd::zero(rid));
                            } else {
                                match serial.send(cmd).await {
                                    Ok(_) => {}
                                    Err(err) => {
                                        tracing::error!("Failed to send player cmd to serial: {}", err);
                                    }
                                }
                            }
                        } else {
                            tracing::error!("Received player cmd but serial is not configured");
                        }
                    }
                    Ok(RuntimeEvent::Debug { msg }) => {
                        tracing::debug!("Runtime debug: {}", msg);
                    }
                    Ok(RuntimeEvent::Crash { msg }) => {
                        tracing::error!("Runtime crash: {}", msg);
                        break;
                    }
                    Ok(RuntimeEvent::Ping) => {
                        tracing::debug!("Runtime ping");
                    }
                    Err(err) => {
                        tracing::error!("Failed to receive runtime msg: {}", err);
                        break;
                    }
                }
            }
        }
    }

    println!("Exiting executor");
    runtime.send(&RuntimeMsg::Term).await?;
    match runtime.wait_with_timeout(Duration::from_secs(2)).await {
        Ok(true) => {}
        Ok(false) => {
            tracing::error!("Python process did not exit in time, killing");
            runtime.kill();
        }
        Err(err) => {
            tracing::error!("Failed to wait for python process: {}", err);
            runtime.kill();
        }
    }

    // Send stop to all players
    tracing::info!("Sending stop to all players");
    if let Some(serial) = &mut serial {
        for id in robots.iter() {
            let cmd = PlayerCmd::zero(*id);
            // cmd.disarm = true;
            if let Err(err) = serial.send(cmd).await {
                tracing::error!("Failed to send stop to player #{}: {}", id, err);
            }
        }
    }

    if let (Some(webui_sender), Some(webui_handle)) = (webui_sender, webui_handle) {
        drop(webui_sender);
        webui_handle.await?;
    }

    Ok(())
}
