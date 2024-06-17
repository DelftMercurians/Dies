use axum::{
    routing::{get, post},
    Router,
};
use dies_core::{PlayerCmd, PlayerId, PlayerOverrideCommand, SymScenario, WorldData};
use dies_executor::{ControlMsg, ExecutorHandle};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc, watch};
use tower_http::services::ServeDir;
use typeshare::typeshare;

use crate::routes::{self, ServerState};

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
    StartScenario {
        scenario: SymScenario,
    },
    SetPause(bool),
    Stop,
}

enum ExecutorTaskCommand {
    ExecutorCommand(ControlMsg),
    StartScenario { scenario: SymScenario },
}

pub async fn start(mut shutdown_rx: broadcast::Receiver<()>) {
    let (inner_update_tx, inner_update_rx) = watch::channel(None);
    let state = Arc::new(ServerState {
        update_rx: inner_update_rx,
        cmd_sender: ui_command_tx,
    });

    let (executor_tx, executor_rx) = mpsc::unbounded_channel::<ExecutorTaskCommand>();

    // Start the web server
    let mut shutdown_rx2 = shutdown_rx.resubscribe();
    let task = tokio::spawn(async move {
        let path = std::env::current_dir()
            .unwrap()
            .join("crates")
            .join("dies-webui")
            .join("static");
        let serve_dir = ServeDir::new(path);
        let app = Router::new()
            .route("/api/state", get(routes::state))
            .route("/api/command", post(routes::command))
            .route("/api/ws", get(routes::websocket))
            .nest_service("/", serve_dir.clone())
            .fallback_service(serve_dir)
            .with_state(Arc::clone(&state));

        let listener = tokio::net::TcpListener::bind("0.0.0.0:5555").await.unwrap();
        let shutdown_fut = async move {
            let _ = shutdown_rx2.recv().await;
        };
        log::info!("Webui running at http://localhost:5555");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .await
            .unwrap();
    });

    // Start executor
    tokio::spawn(async {
        let mut task_handle = None;
        let mut executor_handle = None;
        while let Some(cmd) = executor_rx.recv().await {
            match cmd {
                ExecutorTaskCommand::ExecutorCommand(_) => todo!(),
                ExecutorTaskCommand::StartScenario { scenario } => todo!(),
            }
        }
    });

    // Receive world updates
    loop {
        tokio::select! {
            _ = shutdown_rx.recv() => break,
            update = update_rx.recv() => {
                if let Ok(update) = update {
                    inner_update_tx.send(Some(update)).unwrap();
                } else {
                    break;
                }
            }
        }
    }

    task.await.expect("Shutting down server task failed");
}
