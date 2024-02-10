use std::{
    collections::HashMap,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;

use dies_core::PlayerCmd;
use dies_python_rt::{PyRuntime, PyRuntimeConfig, RuntimeEvent};
use dies_serial_client::{SerialClient, SerialClientConfig};
use dies_ssl_client::{SslVisionClient, SslVisionClientConfig};
use dies_webui::spawn_webui;
use dies_world::{WorldConfig, WorldTracker};
use tokio_util::sync::CancellationToken;

pub struct ExecutorConfig {
    pub py_config: PyRuntimeConfig,
    pub world_config: WorldConfig,
    pub vision_config: Option<SslVisionClientConfig>,
    pub serial_config: Option<SerialClientConfig>,
}

pub async fn run(config: ExecutorConfig, cancel: CancellationToken) -> Result<()> {
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

    // Launch webui
    let (webui_sender, webui_handle) = spawn_webui();

    let mut fail: HashMap<u32, bool> = HashMap::new();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }
            vision_msg = vision.as_mut().unwrap().recv(), if vision.is_some() => {
                match vision_msg {
                    Ok(vision_msg) => {
                        tracker.update_from_protobuf(&vision_msg);

                        if let Some(world_data) = tracker.get() {
                            // Failsafe: if one of our robots is not detected, we send stop to runtime
                            for player in world_data.own_players.iter() {
                                if  SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as f64 - player.timestamp > 0.5 {
                                    fail.insert(player.id, true);
                                    if let Some(serial) = &mut serial {
                                        log::warn!("Failsafe: sending stop to runtime");
                                        serial.send(PlayerCmd {id:player.id, sx: 0.0, sy: 0.0, w: 0.0 }).await?;
                                    } else {
                                        log::error!("Received player cmd but serial is not configured");
                                    }
                                } else {
                                    fail.insert(player.id, false);
                                }
                            }

                            // Send update to runtime
                            if let Err(err) = runtime.send(&dies_python_rt::RuntimeMsg::World(world_data.clone())).await {
                                log::error!("Failed to send world data to runtime: {}", err);
                            }

                            // Send update to webui
                            if let Err(err) = webui_sender.send(world_data) {
                                log::error!("Failed to send world data to webui: {}", err);
                            }
                        }
                    }
                    Err(err) => {
                        log::error!("Failed to receive vision msg: {}", err);
                    }
                }
            }
            runtime_msg = runtime.recv() => {
                match runtime_msg {
                    Ok(RuntimeEvent::PlayerCmd(cmd)) => {
                        if let Some(serial) = &mut serial {
                            if fail.get(&cmd.id) == Some(&true) {
                                log::error!("Failsafe: not sending player cmd");
                            } else {
                                serial.send(cmd).await?;
                            }
                        } else {
                            log::error!("Received player cmd but serial is not configured");
                        }
                    }
                    Ok(RuntimeEvent::Debug { msg }) => {
                        log::debug!("Runtime debug: {}", msg);
                    }
                    Ok(RuntimeEvent::Crash { msg }) => {
                        log::error!("Runtime crash: {}", msg);
                        break;
                    }
                    Err(err) => {
                        log::error!("Failed to receive runtime msg: {}", err);
                        break;
                    }
                }
            }
            // serial_msg = serial.as_mut().unwrap().recv(), if serial.is_some() => {
            //     match serial_msg {
            //         Ok(serial_msg) => {
            //             log::info!("Received serial msg: {}", serial_msg);
            //         }
            //         Err(err) => {
            //             log::error!("Failed to receive serial msg: {}", err);
            //         }
            //     }
            // }
        }
    }

    webui_handle.await?;
    Ok(())
}
