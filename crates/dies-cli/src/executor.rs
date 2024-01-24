use anyhow::Result;

use dies_python_rt::{PyRuntime, PyRuntimeConfig, RuntimeEvent};
use dies_serial_client::{SerialClient, SerialClientConfig};
use dies_ssl_client::{SslVisionClient, SslVisionClientConfig};
use dies_webui::spawn_webui;
use dies_world::{WorldConfig, WorldTracker};
use tokio::sync::oneshot;

pub struct ExecutorConfig {
    pub py_config: PyRuntimeConfig,
    pub world_config: WorldConfig,
    pub vision_config: Option<SslVisionClientConfig>,
    pub serial_config: Option<SerialClientConfig>,
}

pub async fn run(config: ExecutorConfig, mut should_stop: oneshot::Receiver<()>) -> Result<()> {
    let mut runtime = PyRuntime::new(config.py_config).await?;
    let mut tracker = WorldTracker::new(config.world_config);
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

    loop {
        tokio::select! {
            _ = &mut should_stop => {
                break;
            }
            vision_msg = vision.as_mut().unwrap().recv(), if vision.is_some() => {
                match vision_msg {
                    Ok(vision_msg) => {
                        tracker.update_from_protobuf(&vision_msg);

                        if let Some(world_data) = tracker.get() {
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
                            serial.send(cmd).await?;
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
        }
    }

    webui_handle.await?;
    Ok(())
}
