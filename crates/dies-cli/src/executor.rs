use std::{
    collections::{HashMap, HashSet},
    pin::pin,
    time::Duration,
};

use anyhow::{Context, Result};

use dies_control::TeamController;
use dies_core::{PlayerCmd, WorldData};
use dies_protos::{ssl_gc_referee_message::Referee, ssl_vision_wrapper::SSL_WrapperPacket};
use dies_python_rt::{PyRuntime, PyRuntimeConfig, RuntimeEvent, RuntimeMsg};
use dies_serial_client::{SerialClient, SerialClientConfig};
use dies_simulator::Simulation;
use dies_ssl_client::{VisionClient, VisionClientConfig};
use dies_webui::spawn_webui;
use dies_world::{WorldConfig, WorldTracker};
use nalgebra::Vector2;

pub struct ExecutorConfig {
    pub py_config: Option<PyRuntimeConfig>,
    pub world_config: WorldConfig,
    pub vision_config: VisionClientConfig,
    pub serial_config: Option<SerialClientConfig>,
    pub webui: bool,
    /// Maps vision IDs to robot IDs
    pub robot_ids: HashMap<u32, u32>,
}

/// The central component of the framework. It contains all state and logic needed to
/// run a match -- processing vision and referee messages, executing the strategy, and
/// sending commands to the robots.
///
/// The executor can be used in 3 different regimes: externally driven, automatic, and
/// simulation.
///
/// ## Externally driven
///
/// In this regime, the executor is fed with vision and referee messages using the
/// `update_from_vision_msg` and `update_from_gc_msg` methods and the player commands
/// are retrieved using the `get_player_cmds` method. Messages to the game controller
/// are sent using the handle passed with `set_gc_msg_handler`. Update listeners will
/// not be fired in this mode.
///
/// This regime is useful for unit tests and for playing back logs.
///
/// ```no_run
/// let mut executor = Executor::new();
///
/// loop {
///    // Get the vision and referee messages...
///
///    // Update the executor
///    executor.update_from_vision_msg(vision_msg);
///    executor.update_from_gc_msg(gc_msg);
///
///    // Get the player commands
///    let player_cmds = executor.get_player_cmds();
///    // Do something witht the commands...
/// }
/// ```
///
/// ## Automatic
///
/// In this regime, the executor runs its own event loop, receiving vision and referee
/// messages from the network and sending player commands to the robots.
///
/// This regime is useful for running matches and real-life tests in real time.
///
/// ```no_run
/// let mut executor = Executor::new()
///   .with_ssl_client(ssl_client)
///   .with_bs_client(bs_client);
///
/// let (stop_tx, stop_rx) = tokio::sync::oneshot::channel();
/// tokio::spawn(async { executor.run_real_time(stop_rx).await; });
///
/// // Can use stop_tx to stop the event loop from the outside...
/// ```
///
/// It is possible to listen to updates from outside the executor, for example to update
/// the UI:
///
/// ```no_run
/// executor.set_world_update_listener(|update| println!("{:?}", update))
/// ```
///
/// ## Simulation
///
/// In this regime, the executor receives vision and referee messages from the simulation
/// and sends player commands to the simulation. It can either be run in real time with
/// artifical delays in an internal event loop, or it can be driven externally at a
/// faster-than-real-time rate.
///
/// This regime is useful for testing strategies in a simulated environment.
///
/// ```no_run
/// let mut executor = Executor::new()
///    .with_simulator(simulator);
///
/// // Run in real time
/// executor.run_real_time().await;
///
/// // Run at maximum speed
/// loop {
///     executor.step_simulation();
/// }
/// ```
pub struct Executor {
    tracker: WorldTracker,
    // team_controller: TeamController,
    // gc_client: GcClient,
    ssl_client: Option<VisionClient>,
    bs_client: Option<SerialClient>,
    simulator: Option<Simulation>,
    update_listener: Option<Box<dyn Fn(WorldUpdate) -> Result<()> + Send>>,
}

impl Executor {
    /// Create a new instance of [`Executor`].
    pub fn new() -> Executor {
        Executor {
            tracker: WorldTracker::new(WorldConfig {
                initial_opp_goal_x: 1.0,
                is_blue: true,
            }),
            ssl_client: None,
            bs_client: None,
            simulator: None,
        }
    }

    pub fn with_ssl_client(&mut self, ssl_client: VisionClient) -> &mut Self {
        self.ssl_client = Some(ssl_client);
        self
    }

    pub fn with_bs_client(&mut self, bs_client: SerialClient) -> &mut Self {
        self.bs_client = Some(bs_client);
        self
    }

    pub fn with_simulator(&mut self, simulator: Simulation) -> &mut Self {
        self.simulator = Some(simulator);
        self
    }

    pub fn set_play_dir_x(&mut self, opp_x_sign: f32) {
        self.tracker.set_play_dir_x(opp_x_sign);
    }

    // pub fn set_gc_msg_handler(&mut self, handler: impl Fn(Referee) -> Result<()>) {
    //     todo!()
    // }

    pub fn update_from_vision_msg(&mut self, message: SSL_WrapperPacket) {
        todo!()
    }

    pub fn update_from_gc_msg(&mut self, message: Referee) {
        todo!()
    }

    // pub fn update_from_bs_msg(&mut self, message: RobotMessage) {
    //     todo!()
    // }

    pub fn get_player_cmds(&self) -> Vec<PlayerCmd> {
        todo!()
    }

    pub async fn step(&mut self) {
        todo!()
    }

    pub async fn run(self) {
        todo!()
    }
}

pub async fn run(config: ExecutorConfig) -> Result<()> {
    let mut tracker = WorldTracker::new(config.world_config);
    let mut runtime = if let Some(c) = &config.py_config {
        Some(
            PyRuntime::new(c.clone())
                .await
                .context("Failed to create python runtime")?,
        )
    } else {
        None
    };
    let has_runtime = runtime.is_some();
    let mut vision = VisionClient::new(config.vision_config)
        .await
        .context("Failed to create vision client")?;
    let mut serial = match config.serial_config {
        Some(serial_config) => {
            Some(SerialClient::new(serial_config).context("Failed to create the serial client")?)
        }
        None => {
            println!("Serial is not configured");
            None
        }
    };
    let robot_ids = config.robot_ids;

    let mut team_controller = TeamController::new();

    // Launch webui
    let (webui_sender, mut webui_cmd_rx, webui_handle) = if config.webui {
        let (webui_sender, webui_cmd_rx, webui_handle) = spawn_webui();
        (Some(webui_sender), Some(webui_cmd_rx), Some(webui_handle))
    } else {
        (None, None, None)
    };
    let has_webui = webui_sender.is_some();

    let mut ctrlc = pin!(tokio::signal::ctrl_c());

    let fail: HashMap<u32, bool> = HashMap::new();
    let mut robots: HashSet<u32> = HashSet::new();
    let mut world_data: Option<WorldData> = None;
    loop {
        let runtime_msg_fut = async {
            if let Some(runtime) = &mut runtime {
                runtime.recv().await
            } else {
                Err(anyhow::anyhow!("Runtime is not configured"))
            }
        };
        let webui_cmd_rx_fut = async {
            if let Some(webui_cmd_rx) = &mut webui_cmd_rx {
                webui_cmd_rx.recv().await
            } else {
                unreachable!();
            }
        };

        tokio::select! {
            _ = &mut ctrlc => {
                println!("Received Ctrl-C");
                break;
            }
            cmd = webui_cmd_rx_fut, if has_webui => {
                if let Some(cmd) = cmd {
                    if let Some(serial) = &mut serial {
                        let _ = serial.send(cmd).await;
                    } else {
                        tracing::error!("Received player cmd but serial is not configured");
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
                            // for player in world_data.own_players.iter() {
                            //     if  SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as f64 - player.timestamp > 0.5 {
                            //         if fail.get(&player.id) == Some(&false) {
                            //             fail.insert(player.id, true);
                            //             tracing::warn!("Failsafe: sending stop to robot");
                            //             if let Some(serial) = &mut serial {
                            //                 let cmd = PlayerCmd::zero(*robot_ids.get(&player.id).unwrap_or(&0));
                            //                 serial.send_no_wait(cmd);
                            //             } else {
                            //                 tracing::warn!("Received player cmd but serial is not configured");
                            //             }
                            //         }
                            //     } else {
                            //         fail.insert(player.id, false);
                            //     }
                            // }


                            // // Get player 5 data
                            // if let Some(frame) = vision_msg.detection.as_ref() {
                            //     for player in world_data.own_players.iter() {
                            //         if player.id == 5 {
                            //             if let Some(vision_data) = frame.robots_blue.iter().find(|r| r.robot_id == Some(5)) {
                            //                 to_save.extend(vec![
                            //                     player.timestamp,
                            //                     player.position.x as f64,
                            //                     player.position.y as f64,
                            //                     player.velocity.x as f64,
                            //                     player.velocity.y as f64,
                            //                     vision_data.x.unwrap() as f64,
                            //                     vision_data.y.unwrap() as f64,
                            //                     vision_data.orientation.unwrap() as f64,
                            //                 ]);
                            //             }
                            //         }
                            //     }
                            // }

                            // Send update to runtime
                            if let Some(runtime) = &mut runtime {
                                if let Some(world_data) = &world_data {
                                    let _ = runtime.send(&RuntimeMsg::World(world_data.clone())).await;
                                }
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
            runtime_msg = runtime_msg_fut, if has_runtime => {
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
                                let _ = serial.send(cmd);
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

    println!("Exiting executor");
    if let Some(runtime) = runtime.as_mut() {
        runtime
            .send(&RuntimeMsg::Term)
            .await
            .context("Failed to send the termination message to the runtime")?;
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
