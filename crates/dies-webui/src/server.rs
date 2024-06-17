use axum::{
    routing::{get, post},
    Router,
};
use dies_core::{PlayerCmd, PlayerId, PlayerOverrideCommand, WorldData};
use dies_executor::{ControlMsg, ExecutorHandle};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::{
    sync::{broadcast, mpsc, watch},
    task::JoinHandle,
};
use tower_http::services::ServeDir;
use typeshare::typeshare;

// use crate::routes::{self, ServerState};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
#[typeshare]
pub enum ExecutorStatus {
    None,
    Running,
    Paused,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type", content = "data")]
#[typeshare]
pub enum UiCommand {
    SetManualOverride {
        player_id: PlayerId,
        manual_override: bool,
    },
    OverrideCommand {
        player_id: PlayerId,
        command: PlayerOverrideCommand,
    },
    // StartScenario {
    //     scenario: SymScenario,
    // },
    SetPause(bool),
    Stop,
}

enum ExecutorTaskCommand {
    ExecutorMsg(ControlMsg),
    // StartScenario { scenario: SymScenario },
}

enum ExecutorTaskState {
    Runnning {
        task_handle: JoinHandle<()>,
        executor_handle: ExecutorHandle,
    },
    Idle,
}

struct ExecutorTask {
    state: ExecutorTaskState,
    rx: mpsc::UnboundedReceiver<ExecutorTaskCommand>,
}

impl ExecutorTask {
    fn new() -> Self {
        let (executor_tx, rx) = mpsc::unbounded_channel::<ExecutorTaskCommand>();
        Self {
            state: ExecutorTaskState::Idle,
            rx,
        }
    }

    async fn run(&mut self, mut shutdown_rx: broadcast::Receiver<()>) {
        loop {
            tokio::select! {
                cmd = self.rx.recv() => match cmd {
                    Some(cmd) => self.handle_cmd(cmd).await,
                    None => break
                },
                _ = shutdown_rx.recv() => break
            }
        }
    }

    async fn handle_cmd(&mut self, cmd: ExecutorTaskCommand) {
        match cmd {
            ExecutorTaskCommand::ExecutorMsg(msg) => self.handle_executor_msg(msg),
            // ExecutorTaskCommand::StartScenario { scenario } => todo!(),
        };
    }

    async fn stop_executor(&mut self) {}

    fn start_scenario(&mut self) {}

    async fn handle_executor_msg(&mut self, cmd: ControlMsg) {}
}

pub async fn start(mut shutdown_rx: broadcast::Receiver<()>) {
    // let (inner_update_tx, inner_update_rx) = watch::channel(None);
    // let state = Arc::new(ServerState {
    //     update_rx: inner_update_rx,
    //     cmd_sender: ui_command_tx,
    // });

    // let (executor_tx, executor_rx) = mpsc::unbounded_channel::<ExecutorTaskCommand>();

    // // Start the web server
    // let mut shutdown_rx2 = shutdown_rx.resubscribe();
    // let task = tokio::spawn(async move {
    //     let path = std::env::current_dir()
    //         .unwrap()
    //         .join("crates")
    //         .join("dies-webui")
    //         .join("static");
    //     let serve_dir = ServeDir::new(path);
    //     let app = Router::new()
    //         .route("/api/state", get(routes::state))
    //         .route("/api/command", post(routes::command))
    //         .route("/api/ws", get(routes::websocket))
    //         .nest_service("/", serve_dir.clone())
    //         .fallback_service(serve_dir)
    //         .with_state(Arc::clone(&state));

    //     let listener = tokio::net::TcpListener::bind("0.0.0.0:5555").await.unwrap();
    //     let shutdown_fut = async move {
    //         let _ = shutdown_rx2.recv().await;
    //     };
    //     log::info!("Webui running at http://localhost:5555");
    //     axum::serve(listener, app)
    //         .with_graceful_shutdown(shutdown_fut)
    //         .await
    //         .unwrap();
    // });

    // // Start executor
    // let mut shutdown_rx3 = shutdown_rx.resubscribe();
    // tokio::spawn(async {
    //     let mut handles = None;
    //     loop {
    //         tokio::select! {
    //             cmd = executor_rx.recv() => {
    //                 match cmd {
    //                    Some(ExecutorTaskCommand::ExecutorMsg(_)) => todo!(),
    //                    Some(ExecutorTaskCommand::StartScenario { scenario }) => todo!()
    //                 }
    //             }
    //         }
    //     }
    //     while let Some(cmd) = executor_rx.recv().await {
    //         match cmd {
    //             ExecutorTaskCommand::ExecutorMsg(_) => todo!(),
    //             ExecutorTaskCommand::StartScenario { scenario } => todo!(),
    //         }
    //     }
    // });

    // // Receive world updates
    // loop {
    //     tokio::select! {
    //         _ = shutdown_rx.recv() => break,
    //         update = update_rx.recv() => {
    //             if let Ok(update) = update {
    //                 inner_update_tx.send(Some(update)).unwrap();
    //             } else {
    //                 break;
    //             }
    //         }
    //     }
    // }

    // task.await.expect("Shutting down server task failed");
}
