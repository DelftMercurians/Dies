use std::sync::Arc;

use anyhow::anyhow;
use dies_core::WorldUpdate;
use dies_executor::{scenarios::ScenarioType, ControlMsg, ExecutorHandle};
use dies_protos::ssl_gc_referee_message::referee::Command;
use dies_simulator::SimulationConfig;
use tokio::{
    sync::{broadcast, oneshot, watch},
    task::JoinHandle,
};

use crate::{server::ServerState, ExecutorStatus, UiCommand, UiEnvironment, UiMode};

#[derive(Default)]
enum ExecutorTaskState {
    #[default]
    Idle,
    Starting {
        task_handle: JoinHandle<()>,
        cancel_tx: oneshot::Sender<()>,
    },
    Runnning {
        task_handle: JoinHandle<()>,
        executor_handle: ExecutorHandle,
    },
}

pub struct ExecutorTask {
    state: ExecutorTaskState,
    cmd_rx: broadcast::Receiver<UiCommand>,
    update_tx: watch::Sender<Option<WorldUpdate>>,
    server_state: Arc<ServerState>,
    ui_env: UiEnvironment,
    sim_config: SimulationConfig,
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
            sim_config: SimulationConfig::default(),
        }
    }

    pub async fn run(&mut self, mut shutdown_rx: broadcast::Receiver<()>) {
        loop {
            let shutdown_rx2 = shutdown_rx.resubscribe();
            tokio::select! {
                cmd = self.cmd_rx.recv() => match cmd {
                    Ok(cmd) => self.handle_cmd(cmd, shutdown_rx2).await,
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

    async fn handle_cmd(&mut self, cmd: UiCommand, mut shutdown_rx: broadcast::Receiver<()>) {
        match cmd {
            UiCommand::SetManualOverride {
                player_id,
                manual_override,
            } => self.handle_executor_msg(ControlMsg::SetPlayerOverride {
                player_id,
                override_active: manual_override,
            }),
            UiCommand::OverrideCommand { player_id, command } => {
                self.handle_executor_msg(ControlMsg::PlayerOverrideCommand(player_id, command))
            }
            UiCommand::SimulatorCmd(cmd) => self.handle_executor_msg(ControlMsg::SimulatorCmd(cmd)),
            UiCommand::SetPause(pause) => self.handle_executor_msg(ControlMsg::SetPause(pause)),
            UiCommand::Stop => self.stop_executor().await,
            UiCommand::StartScenario { scenario } => {
                log::info!("Starting scenarion \"{}\"", scenario.name());
                // This is a potentially long running operation (in case of live mode),
                // so we allow cancelling
                tokio::select! {
                    _ = shutdown_rx.recv() => self.stop_executor().await,
                    _ = self.start_scenario(scenario) => {}
                }
            }
            UiCommand::GcCommand(command) => {
                let cmd = string_to_command(command).unwrap();
                self.handle_executor_msg(ControlMsg::GcCommand { command: cmd });
            }
        }
    }

    async fn stop_executor(&mut self) {
        match std::mem::take(&mut self.state) {
            ExecutorTaskState::Runnning {
                task_handle,
                executor_handle,
            } => {
                executor_handle.send(ControlMsg::Stop);
                let _ = task_handle.await;
                log::info!("Executor stopped");
            }
            ExecutorTaskState::Starting {
                cancel_tx,
                task_handle,
            } => {
                let _ = cancel_tx.send(());
                let _ = task_handle.await;
                log::info!("Executor startup cancelled");
            }
            ExecutorTaskState::Idle => {}
        }
        self.server_state.set_executor_status(ExecutorStatus::None);
    }

    async fn start_scenario(&mut self, scenario: ScenarioType) {
        let setup = scenario.into_setup();
        let mode = {
            let mode = self.server_state.ui_mode.read().unwrap();
            *mode
        };

        let (handle_tx, handle_rx) = oneshot::channel::<ExecutorHandle>();
        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
        let task_handle = {
            let settings = { self.server_state.executor_settings.read().unwrap().clone() };
            let ui_env = self.ui_env.clone();
            let sim_config = self.sim_config.clone();
            let server_state = Arc::clone(&self.server_state);
            let update_tx = self.update_tx.clone();
            let res = tokio::spawn(async move {
                let log_file_name = {
                    let time = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string();
                    format!("dies-{time}.log")
                };
                dies_core::debug_clear();
                dies_logger::log_start(log_file_name);

                let executor = match (mode, ui_env) {
                    (UiMode::Simulation, _) => Ok(setup.into_simulation(settings, sim_config)),
                    (
                        UiMode::Live,
                        UiEnvironment::WithLive {
                            ssl_config,
                            bs_handle,
                        },
                    ) => {
                        log::info!("Starting live scenario {}", scenario.name());
                        server_state.set_executor_status(ExecutorStatus::StartingScenario(
                            setup.get_info(),
                        ));

                        tokio::select! {
                            _ = cancel_rx => Err(anyhow::anyhow!("Cancelled")),
                            executor = setup.into_live(settings, ssl_config, bs_handle) => executor
                        }
                    }
                    (UiMode::Live, UiEnvironment::SimulationOnly) => {
                        Err(anyhow::anyhow!("Live mode not available"))
                    }
                };

                match executor {
                    Ok(executor) => {
                        log::info!("Scenario started, executor ready");

                        // Relay world update to the UI
                        let _ = handle_tx.send(executor.handle());
                        let mut handle = executor.handle();
                        tokio::spawn(async move {
                            while let Ok(update) = handle.update_rx.recv().await {
                                let _ = update_tx.send(Some(update));
                            }
                        });

                        server_state.set_executor_status(ExecutorStatus::RunningExecutor {
                            scenario: scenario.name().to_owned(),
                        });
                        if let Err(err) = executor.run_real_time().await {
                            log::error!("Executor failed: {}", err);
                        }
                    }
                    Err(err) => {
                        log::error!("Failed to start scenario: {}", err);
                        server_state
                            .set_executor_status(ExecutorStatus::Failed(format!("{}", err)));
                    }
                }

                dies_logger::log_close();
            });

            tokio::spawn(async {
                match res.await {
                    Ok(_) => {}
                    Err(err) => {
                        log::error!("Executor task failed: {}", err);
                        std::process::exit(1);
                    }
                }
            })
        };

        self.state = ExecutorTaskState::Starting {
            task_handle,
            cancel_tx,
        };

        if let Ok(executor_handle) = handle_rx.await {
            if let ExecutorTaskState::Starting { task_handle, .. } = std::mem::take(&mut self.state)
            {
                *self.server_state.executor_handle.write().unwrap() = Some(executor_handle.clone());
                self.state = ExecutorTaskState::Runnning {
                    task_handle,
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

fn string_to_command(command_str: String) -> anyhow::Result<Command> {
    match command_str.as_str() {
        "HALT" => Ok(Command::HALT),
        "STOP" => Ok(Command::STOP),
        "NORMAL_START" => Ok(Command::NORMAL_START),
        "FORCE_START" => Ok(Command::FORCE_START),
        "PREPARE_KICKOFF_YELLOW" => Ok(Command::PREPARE_KICKOFF_YELLOW),
        "PREPARE_KICKOFF_BLUE" => Ok(Command::PREPARE_KICKOFF_BLUE),
        "PREPARE_PENALTY_YELLOW" => Ok(Command::PREPARE_PENALTY_YELLOW),
        "PREPARE_PENALTY_BLUE" => Ok(Command::PREPARE_PENALTY_BLUE),
        "DIRECT_FREE_YELLOW" => Ok(Command::DIRECT_FREE_YELLOW),
        "DIRECT_FREE_BLUE" => Ok(Command::DIRECT_FREE_BLUE),
        "INDIRECT_FREE_YELLOW" => Ok(Command::INDIRECT_FREE_YELLOW),
        "INDIRECT_FREE_BLUE" => Ok(Command::INDIRECT_FREE_BLUE),
        "TIMEOUT_YELLOW" => Ok(Command::TIMEOUT_YELLOW),
        "TIMEOUT_BLUE" => Ok(Command::TIMEOUT_BLUE),
        "GOAL_YELLOW" => Ok(Command::GOAL_YELLOW),
        "GOAL_BLUE" => Ok(Command::GOAL_BLUE),
        "BALL_PLACEMENT_YELLOW" => Ok(Command::BALL_PLACEMENT_YELLOW),
        "BALL_PLACEMENT_BLUE" => Ok(Command::BALL_PLACEMENT_BLUE),
        _ => Err(anyhow!("Unknown command: {}", command_str)),
    }
}
