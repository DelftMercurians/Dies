use std::{
    collections::{HashMap, HashSet},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use npy::to_file;

use dies_control::TeamController;
use dies_core::{PlayerCmd, WorldData};
use dies_python_rt::{PyRuntime, PyRuntimeConfig, RuntimeEvent, RuntimeMsg};
use dies_serial_client::{SerialClient, SerialClientConfig};
use dies_ssl_client::{VisionClient, VisionClientConfig};
use dies_webui::spawn_webui;
use dies_world::{WorldConfig, WorldTracker};
use nalgebra::Vector2;

pub struct ExecutorConfig {
    pub py_config: PyRuntimeConfig,
    pub world_config: WorldConfig,
    pub vision_config: VisionClientConfig,
    pub serial_config: Option<SerialClientConfig>,
    pub webui: bool,
    /// Maps vision IDs to robot IDs
    pub robot_ids: HashMap<u32, u32>,
}

pub async fn run(config: ExecutorConfig) -> Result<()> {
    let mut tracker = WorldTracker::new(config.world_config);
    let mut runtime = PyRuntime::new(config.py_config).await?;
    let mut vision = VisionClient::new(config.vision_config)
        .await
        .context("Failed to create vision client")?;
    let mut serial = match config.serial_config {
        Some(serial_config) => {
            Some(SerialClient::new(serial_config).context("Failed to create the serial client")?)
        }
        None => None,
    };
    let robot_ids = config.robot_ids;

    let mut team_controller = TeamController::new();

    // Launch webui
    let (webui_sender, webui_handle) = if config.webui {
        let (webui_sender, webui_handle) = spawn_webui();
        (Some(webui_sender), Some(webui_handle))
    } else {
        (None, None)
    };

    let mut fail: HashMap<u32, bool> = HashMap::new();
    let mut robots: HashSet<u32> = HashSet::new();
    let mut update_interval = tokio::time::interval(Duration::from_millis(50));
    let mut world_data: Option<WorldData> = None;
    loop {
        tokio::select! {
            _ = update_interval.tick() => {
                if let Some(world_data) = &world_data {
                    let cmds = team_controller.update(&world_data.own_players);
                    if let Some(serial) = &mut serial {
                        for mut cmd in cmds {
                            let rid = *robot_ids.get(&cmd.id).unwrap_or(&0);
                            cmd.id = rid;
                            match serial.send(cmd).await {
                                Ok(_) => {}
                                Err(err) => {
                                    tracing::error!("Failed to send player cmd to serial: {}", err);
                                }
                            }
                        }
                    }
                }
            }
            vision_msg = vision.recv() => {
                match vision_msg {
                    Ok(vision_msg) => {
                        tracker.update_from_vision(&vision_msg);
                        if let Some(new_world_data) = tracker.get() {
                            world_data = Some(new_world_data.clone());
                            // Failsafe: if one of our robots is not detected, we send stop to runtime
                            for player in new_world_data.own_players.iter() {
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


                            // Get player 5 data
                            if let Some(frame) = vision_msg.detection.as_ref() {
                                for player in world_data.own_players.iter() {
                                    if player.id == 5 {
                                        if let Some(vision_data) = frame.robots_blue.iter().find(|r| r.robot_id == Some(5)) {
                                            to_save.extend(vec![
                                                player.timestamp,
                                                player.position.x as f64,
                                                player.position.y as f64,
                                                player.velocity.x as f64,
                                                player.velocity.y as f64,
                                                vision_data.x.unwrap() as f64,
                                                vision_data.y.unwrap() as f64,
                                                vision_data.orientation.unwrap() as f64,
                                            ]);
                                        }
                                    }
                                }
                            }

                            // Send update to runtime
                            if let Err(err) = runtime.send(&dies_python_rt::RuntimeMsg::World(new_world_data.clone())).await {
                                tracing::error!("Failed to send world data to runtime: {}", err);
                            }

                            // Send update to webui
                            if let Some(ref webui_sender) = webui_sender {
                                if let Err(err) = webui_sender.send(new_world_data) {
                                    tracing::error!("Failed to send world data to webui: {}", err);
                                }
                            }
                        }
                    }
                    Err(err) => {
                        tracing::error!("Failed to receive vision msg: {}", err);
                    }
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
                    Ok(RuntimeEvent::PlayerPosCmd(cmd)) => {
                        let rid = *robot_ids.get(&cmd.id).unwrap_or(&0);
                        robots.insert(rid);
                        team_controller.set_target_pos(cmd.id, Vector2::new(cmd.x, cmd.y));
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

    let _ = runtime
        .send(&RuntimeMsg::Term)
        .await
        .map_err(|e| tracing::error!("Failed to send term to runtime: {}", e));
    match runtime.wait_with_timeout(Duration::from_secs(2)).await {
        Ok(true) => {}
        Ok(false) => {
            tracing::error!("Python process did not exit in time, killing");
            runtime.kill();
        }
        Err(err) => {
            tracing::error!("Failed to wait for python process, killing: {}", err);
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
        webui_handle
            .await
            .context("Failed joining the webui task")?;
    }

    Ok(())
}
