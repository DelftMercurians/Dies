use std::sync::Arc;

use anyhow::anyhow;
use dies_core::{SideAssignment, WorldUpdate};
use dies_executor::{ControlMsg, Executor, ExecutorHandle, Strategy};
use dies_protos::ssl_gc_referee_message::referee::Command;
use dies_simulator::SimulationBuilder;
use dies_ssl_client::VisionClient;
use tokio::{
    sync::{broadcast, oneshot, watch},
    task::JoinHandle,
};

use crate::{server::ServerState, ExecutorStatus, UiCommand, UiEnvironment, UiMode};

#[derive(Default)]
enum ExecutorTaskState {
    #[default]
    Idle,
    Starting,
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
    strategy: Strategy,
}

impl ExecutorTask {
    pub fn new(
        config: UiEnvironment,
        update_tx: watch::Sender<Option<WorldUpdate>>,
        cmd_rx: broadcast::Receiver<UiCommand>,
        server_state: Arc<ServerState>,
        strategy: Strategy,
    ) -> Self {
        Self {
            state: ExecutorTaskState::Idle,
            update_tx,
            cmd_rx,
            server_state,
            ui_env: config,
            strategy,
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
            UiCommand::SetTeamScriptPaths {
                blue_script_path,
                yellow_script_path,
            } => {
                // Update team configuration in settings
                {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    settings.team_configuration.blue_script_path = blue_script_path.clone();
                    settings.team_configuration.yellow_script_path = yellow_script_path.clone();
                }
                self.handle_executor_msg(ControlMsg::SetTeamScriptPaths {
                    blue_script_path,
                    yellow_script_path,
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
                let (new_blue_active, new_yellow_active, new_blue_script, new_yellow_script) = {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    // Swap active teams
                    let old_blue_active = settings.team_configuration.blue_active;
                    let old_yellow_active = settings.team_configuration.yellow_active;
                    settings.team_configuration.blue_active = old_yellow_active;
                    settings.team_configuration.yellow_active = old_blue_active;

                    // Swap script paths
                    let old_blue_script = settings.team_configuration.blue_script_path.clone();
                    let old_yellow_script = settings.team_configuration.yellow_script_path.clone();
                    settings.team_configuration.blue_script_path = old_yellow_script.clone();
                    settings.team_configuration.yellow_script_path = old_blue_script.clone();

                    (
                        old_yellow_active,
                        old_blue_active,
                        old_yellow_script,
                        old_blue_script,
                    )
                };

                self.handle_executor_msg(ControlMsg::SwapTeamColors);
            }
            UiCommand::SwapTeamSides => {
                let new_side_assignment = {
                    let mut settings = self.server_state.executor_settings.write().unwrap();
                    let new_assignment = match settings.team_configuration.side_assignment {
                        SideAssignment::BlueOnPositive => SideAssignment::YellowOnPositive,
                        SideAssignment::YellowOnPositive => SideAssignment::BlueOnPositive,
                    };
                    settings.team_configuration.side_assignment = new_assignment;
                    new_assignment
                };

                self.handle_executor_msg(ControlMsg::SwapTeamSides);
            }
            UiCommand::SimulatorCmd(cmd) => self.handle_executor_msg(ControlMsg::SimulatorCmd(cmd)),
            UiCommand::SetPause(pause) => self.handle_executor_msg(ControlMsg::SetPause(pause)),
            UiCommand::Stop => self.stop_executor().await,
            UiCommand::Start => {
                log::info!("Starting executor");
                self.start(self.strategy).await;
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
            ExecutorTaskState::Starting => {}
            ExecutorTaskState::Idle => {}
        }
        self.server_state.set_executor_status(ExecutorStatus::None);
    }

    async fn start(&mut self, strategy: Strategy) {
        let mode = {
            let mode = self.server_state.ui_mode.read().unwrap();
            *mode
        };

        let (handle_tx, handle_rx) = oneshot::channel::<ExecutorHandle>();
        let task_handle = {
            let settings = { self.server_state.executor_settings.read().unwrap().clone() };
            let ui_env = self.ui_env.clone();
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
                    (UiMode::Simulation, _) => Ok(Executor::new_simulation(
                        settings,
                        SimulationBuilder::default().build(),
                        strategy,
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
                            Ok(Executor::new_live(
                                settings,
                                vision_client,
                                bs_handle,
                                strategy,
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

                        // Relay world update to the UI
                        let _ = handle_tx.send(executor.handle());
                        let mut handle = executor.handle();

                        // Spawn task to relay world updates
                        let update_tx_clone = update_tx.clone();
                        tokio::spawn(async move {
                            while let Ok(update) = handle.update_rx.recv().await {
                                let _ = update_tx_clone.send(Some(update));
                            }
                        });

                        // Spawn task to relay script errors via WebSocket
                        let server_state_clone = Arc::clone(&server_state);
                        let mut handle_clone = executor.handle();
                        tokio::spawn(async move {
                            while let Some(script_error) = handle_clone.recv_script_error().await {
                                // Broadcast script error through the server state
                                if let Err(err) =
                                    server_state_clone.script_error_tx.send(script_error)
                                {
                                    log::error!("Failed to broadcast script error: {}", err);
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

        self.state = ExecutorTaskState::Starting;
        if let Ok(executor_handle) = handle_rx.await {
            if let ExecutorTaskState::Starting = std::mem::take(&mut self.state) {
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
