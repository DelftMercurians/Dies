use std::sync::Arc;
use std::thread;

use dies_core::WorldUpdate;
use dies_executor::{ControlMsg, Executor, ExecutorHandle};
use dies_simulator::SimulationBuilder;
use dies_ssl_client::{SslClientConfig, VisionClient};
use tokio::sync::{
    broadcast::{self, error::RecvError},
    oneshot, watch,
};

use crate::{
    replay_controller::ReplayController, server::ServerState, ExecutorStatus, UiCommand,
    UiEnvironment, UiMode,
};

#[derive(Default)]
enum ExecutorTaskState {
    #[default]
    Idle,
    Starting,
    Runnning {
        thread_handle: thread::JoinHandle<()>,
        executor_handle: ExecutorHandle,
    },
    Replaying {
        controller: ReplayController,
    },
}

pub struct ExecutorTask {
    state: ExecutorTaskState,
    cmd_rx: broadcast::Receiver<UiCommand>,
    update_tx: watch::Sender<Option<WorldUpdate>>,
    server_state: Arc<ServerState>,
    ui_env: UiEnvironment,
}

impl ExecutorTask {
    pub fn new(
        config: UiEnvironment,
        update_tx: watch::Sender<Option<WorldUpdate>>,
        cmd_rx: broadcast::Receiver<UiCommand>,
        server_state: Arc<ServerState>,
    ) -> Self {
        Self {
            state: ExecutorTaskState::Idle,
            update_tx,
            cmd_rx,
            server_state,
            ui_env: config,
        }
    }

    pub async fn run(&mut self, mut shutdown_rx: broadcast::Receiver<()>) {
        loop {
            tokio::select! {
                cmd = self.cmd_rx.recv() => match cmd {
                    Ok(cmd) => self.handle_cmd(cmd).await,
                    Err(err) => match err {
                        broadcast::error::RecvError::Lagged(_) => {}
                        broadcast::error::RecvError::Closed => break
                    }
                },
                _ = shutdown_rx.recv() => {
                    self.stop_executor().await;
                    break;
                }
            }
        }
    }

    async fn handle_cmd(&mut self, cmd: UiCommand) {
        match cmd {
            UiCommand::SetManualOverride {
                team_color,
                player_id,
                manual_override,
            } => self.handle_executor_msg(ControlMsg::SetPlayerOverride {
                team_color,
                player_id,
                override_active: manual_override,
            }),
            UiCommand::OverrideCommand {
                team_color,
                player_id,
                command,
            } => self.handle_executor_msg(ControlMsg::PlayerOverrideCommand {
                team_color,
                player_id,
                command,
            }),
            UiCommand::SetActiveTeams {
                blue_active,
                yellow_active,
            } => {
                // Update team configuration in settings
                {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    settings.team_configuration.blue_active = blue_active;
                    settings.team_configuration.yellow_active = yellow_active;
                }
                self.handle_executor_msg(ControlMsg::SetActiveTeams {
                    blue_active,
                    yellow_active,
                });
            }
            UiCommand::SetSideAssignment { side_assignment } => {
                // Update team configuration in settings
                {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    settings.team_configuration.side_assignment = side_assignment;
                }
                self.handle_executor_msg(ControlMsg::SetSideAssignment(side_assignment));
            }
            UiCommand::SetTeamConfiguration { configuration } => {
                // Update team configuration in settings
                {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    settings.team_configuration = configuration.clone();
                }
                self.handle_executor_msg(ControlMsg::SetTeamConfiguration(configuration));
            }
            UiCommand::SwapTeamColors => {
                self.handle_executor_msg(ControlMsg::SwapTeamColors);
            }
            UiCommand::SwapTeamSides => {
                self.handle_executor_msg(ControlMsg::SwapTeamSides);
            }
            UiCommand::SimulatorCmd(cmd) => self.handle_executor_msg(ControlMsg::SimulatorCmd(cmd)),
            UiCommand::SetPause(pause) => self.handle_executor_msg(ControlMsg::SetPause(pause)),
            UiCommand::Stop => self.stop_executor().await,
            UiCommand::Start => {
                log::info!("Starting executor");
                self.start().await;
            }
            UiCommand::GcCommand(command) => {
                self.handle_executor_msg(ControlMsg::GcCommand { command });
            }
            UiCommand::AddMarker { label } => {
                self.handle_executor_msg(ControlMsg::AddMarker { label });
            }
            UiCommand::LoadLog { path } => self.load_replay(path).await,
            UiCommand::ReplayPlay => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.play();
                }
            }
            UiCommand::ReplayPause => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.pause();
                }
            }
            UiCommand::ReplaySeek { t } => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.seek(t);
                }
            }
            UiCommand::ReplaySetSpeed { speed } => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.set_speed(speed);
                }
            }
            UiCommand::ReplayStep { delta } => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.step(delta);
                }
            }
            UiCommand::ReplayStepTime { dt } => {
                if let ExecutorTaskState::Replaying { controller } = &self.state {
                    controller.step_time(dt);
                }
            }
            UiCommand::SetStrategyParam {
                team_color,
                key,
                value,
            } => {
                let mut params = std::collections::HashMap::new();
                params.insert(key, value);
                self.handle_executor_msg(ControlMsg::SetStrategyParams {
                    team: team_color,
                    params,
                });
            }
            // Handled by the dedicated bench task (direct-to-basestation), not
            // the executor.
            UiCommand::Bench(_) => {}
        }
    }

    /// Stop any running executor/replay and load a recorded log for replay.
    async fn load_replay(&mut self, path: String) {
        self.stop_executor().await;
        match ReplayController::load(
            &path,
            self.update_tx.clone(),
            self.server_state.replay_state.clone(),
        ) {
            Ok(controller) => {
                log::info!("Loaded replay log: {path}");
                self.state = ExecutorTaskState::Replaying { controller };
            }
            Err(err) => log::error!("Failed to load replay log {path}: {err}"),
        }
    }

    async fn stop_executor(&mut self) {
        match std::mem::take(&mut self.state) {
            ExecutorTaskState::Runnning {
                thread_handle,
                executor_handle,
            } => {
                log::info!("Shutdown: sending Stop to executor, joining executor thread");
                executor_handle.send(ControlMsg::Stop);
                // Join the thread on a blocking helper so we don't stall the
                // tokio runtime while the executor wraps up.
                let _ = tokio::task::spawn_blocking(move || thread_handle.join()).await;
                log::info!("Executor stopped");
            }
            ExecutorTaskState::Replaying { controller } => {
                controller.stop(&self.server_state.replay_state);
                log::info!("Replay stopped");
            }
            ExecutorTaskState::Starting => {}
            ExecutorTaskState::Idle => {}
        }
        self.server_state.set_executor_status(ExecutorStatus::None);
    }

    async fn start(&mut self) {
        let mode = {
            let mode = self.server_state.ui_mode.read().unwrap();
            *mode
        };

        let (handle_tx, handle_rx) = oneshot::channel::<ExecutorHandle>();
        let settings = { self.server_state.executor_settings.read().unwrap().clone() };
        let ui_env = self.ui_env.clone();
        let server_state = Arc::clone(&self.server_state);
        let update_tx = self.update_tx.clone();

        // Run the executor on a dedicated OS thread with its own single-threaded
        // tokio runtime. This keeps the !Send Executor (when in test mode, it
        // holds a Boa JS context) off the main multi-threaded runtime without
        // sprinkling LocalSet everywhere.
        let thread_handle = thread::Builder::new()
            .name("dies-executor".into())
            .spawn(move || {
                let rt = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(rt) => rt,
                    Err(err) => {
                        log::error!("Failed to build executor runtime: {}", err);
                        return;
                    }
                };

                rt.block_on(async move {
                    let session_name = {
                        let time = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                        format!("dies-{time}")
                    };
                    dies_core::debug_clear();
                    let mut meta = dies_logger::MetaJson::new(
                        chrono::Utc::now().timestamp_millis() as f64 / 1000.0,
                        matches!(mode, UiMode::Simulation),
                        settings.team_configuration.blue_strategy.clone(),
                        settings.team_configuration.yellow_strategy.clone(),
                        dies_logger::side_assignment_str(
                            settings.team_configuration.side_assignment,
                        )
                        .to_string(),
                    );
                    meta.is_match = settings.is_match;
                    dies_logger::worker::log_start(&session_name, meta);
                    dies_logger::worker::log_settings_baseline(0, &settings);

                    let executor = match (mode, ui_env) {
                        (UiMode::Simulation, _) => Ok(Executor::new_simulation(
                            settings,
                            SimulationBuilder::default().build(),
                        )),
                        (
                            UiMode::Live,
                            UiEnvironment::WithLive {
                                ssl_config,
                                bs_handle,
                            },
                        ) => {
                            // Try the configured vision client (if any). `None`
                            // here means no vision was configured at launch.
                            let vision_client = match ssl_config {
                                Some(cfg) => VisionClient::new(cfg).await.ok(),
                                None => None,
                            };
                            if let Some(vision_client) = vision_client {
                                Ok(Executor::new_live(settings, vision_client, bs_handle))
                            } else if settings.allow_no_vision {
                                log::warn!("Starting executor with mock vision client");
                                Ok(Executor::new_live(
                                    settings,
                                    VisionClient::new(SslClientConfig {
                                        vision: dies_ssl_client::ConnectionConfig::Mock,
                                        gc: dies_ssl_client::ConnectionConfig::Mock,
                                    })
                                    .await
                                    .unwrap(),
                                    bs_handle,
                                ))
                            } else {
                                Err(anyhow::anyhow!(
                                    "No vision available (configure --connection-mode or enable allow_no_vision)"
                                ))
                            }
                        }
                        (UiMode::Live, UiEnvironment::SimulationOnly) => {
                            Err(anyhow::anyhow!("Live mode not available"))
                        }
                    };

                    match executor {
                        Ok(executor) => {
                            log::info!("Executor ready");
                            let _ = handle_tx.send(executor.handle());
                            let mut handle = executor.handle();

                            let update_tx_clone = update_tx.clone();
                            tokio::spawn(async move {
                                loop {
                                    match handle.update_rx.recv().await {
                                        Ok(update) => {
                                            let _ = update_tx_clone.send(Some(update));
                                        }
                                        Err(RecvError::Lagged(_)) => continue,
                                        Err(RecvError::Closed) => {
                                            log::info!("Executor update channel closed");
                                            break;
                                        }
                                    }
                                }
                            });

                            server_state.set_executor_status(ExecutorStatus::RunningExecutor);
                            if let Err(err) = executor.run_real_time().await {
                                log::error!("Executor failed: {}", err);
                            }
                        }
                        Err(err) => {
                            log::error!("Failed to start executor: {}", err);
                            server_state
                                .set_executor_status(ExecutorStatus::Failed(format!("{}", err)));
                        }
                    }

                    // Block so the process doesn't exit mid-compaction.
                    log::info!(
                        "Shutdown: executor run loop ended, closing log (blocking, up to 25s)"
                    );
                    dies_logger::worker::log_close_blocking(std::time::Duration::from_secs(25));
                    log::info!("Shutdown: log close complete, executor thread exiting");
                });
            })
            .expect("spawn executor thread");

        self.state = ExecutorTaskState::Starting;
        if let Ok(executor_handle) = handle_rx.await {
            if let ExecutorTaskState::Starting = std::mem::take(&mut self.state) {
                *self.server_state.executor_handle.write().unwrap() = Some(executor_handle.clone());
                self.state = ExecutorTaskState::Runnning {
                    thread_handle,
                    executor_handle,
                };
            } else {
                log::warn!("Received executor handle but state is not starting");
            }
        } else {
            self.state = ExecutorTaskState::Idle;
        }
    }

    fn handle_executor_msg(&mut self, cmd: ControlMsg) {
        if let ExecutorTaskState::Runnning {
            executor_handle, ..
        } = &mut self.state
        {
            executor_handle.send(cmd);
        }
    }
}
