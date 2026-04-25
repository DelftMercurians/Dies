use std::path::PathBuf;
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

use crate::{server::ServerState, ExecutorStatus, UiCommand, UiEnvironment, UiMode};

#[derive(Default)]
enum ExecutorTaskState {
    #[default]
    Idle,
    Starting,
    Runnning {
        thread_handle: thread::JoinHandle<()>,
        executor_handle: ExecutorHandle,
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
            UiCommand::StartScenario { scenario, team } => {
                let path = resolve_scenario_path(&scenario);
                if !path.exists() {
                    log::warn!("scenario file not found: {}", path.display());
                    return;
                }
                self.handle_executor_msg(ControlMsg::StartScenario { path, team });
            }
            UiCommand::StopScenario => {
                self.handle_executor_msg(ControlMsg::StopScenario);
            }
        }
    }

    async fn stop_executor(&mut self) {
        match std::mem::take(&mut self.state) {
            ExecutorTaskState::Runnning {
                thread_handle,
                executor_handle,
            } => {
                executor_handle.send(ControlMsg::Stop);
                // Join the thread on a blocking helper so we don't stall the
                // tokio runtime while the executor wraps up.
                let _ = tokio::task::spawn_blocking(move || thread_handle.join()).await;
                log::info!("Executor stopped");
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
                    let log_file_name = {
                        let time = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                        format!("dies-{time}.log")
                    };
                    dies_core::debug_clear();
                    dies_logger::log_start(log_file_name);

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
                            let vision_client = VisionClient::new(ssl_config).await;
                            if let Ok(vision_client) = vision_client {
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
                                Err(anyhow::anyhow!("Failed to connect to vision"))
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

                            // Bridge scenario log entries into the long-lived
                            // server-side broadcast so WS clients don't need to
                            // resubscribe each Start.
                            let mut log_rx = executor.handle().log_bus.subscribe();
                            let log_tx = server_state.scenario_log_tx.clone();
                            tokio::spawn(async move {
                                loop {
                                    match log_rx.recv().await {
                                        Ok(entry) => {
                                            let _ = log_tx.send(entry);
                                        }
                                        Err(RecvError::Lagged(_)) => continue,
                                        Err(RecvError::Closed) => break,
                                    }
                                }
                            });

                            // Mirror scenario status into server state.
                            let mut status_rx = executor.handle().scenario_status_rx.clone();
                            let status_tx = server_state.scenario_status.clone();
                            tokio::spawn(async move {
                                let _ = status_tx.send(status_rx.borrow().clone());
                                while status_rx.changed().await.is_ok() {
                                    let _ = status_tx.send(status_rx.borrow().clone());
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

                    dies_logger::log_close();
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

/// Resolve a scenario name from the UI to an on-disk path. The UI sends bare
/// names (e.g. `mpc_step_response.js`); they are resolved relative to the
/// `scenarios/` directory in the current working directory. Absolute paths
/// and explicit relative paths (containing `/`) are kept as-is.
pub(crate) fn resolve_scenario_path(name: &str) -> PathBuf {
    let p = PathBuf::from(name);
    if p.is_absolute() || name.contains('/') || name.contains(std::path::MAIN_SEPARATOR) {
        return p;
    }
    let mut full = PathBuf::from("scenarios");
    full.push(name);
    full
}
