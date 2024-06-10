use axum::{
    routing::{get, post},
    Router,
};
use dies_core::{PlayerCmd, SymScenario};
use dies_executor::WorldUpdate;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{broadcast, watch};
use tower_http::services::ServeDir;

use crate::routes;

#[derive(Debug, Clone, Serialize)]
pub struct UiSettings {
    pub can_control: bool,
}

pub(crate) struct ServerState {
    pub(crate) settings: UiSettings,
    pub(crate) world_state: watch::Receiver<Option<WorldUpdate>>,
    pub(crate) cmd_sender: broadcast::Sender<UiCommand>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", tag = "type")]
pub enum UiCommand {
    DirectPlayerCmd { cmd: PlayerCmd },
    SelectScenarioCmd { scenario: SymScenario },
    StartCmd,
    StopCmd,
}

pub async fn start(
    settings: UiSettings,
    mut update_rx: broadcast::Receiver<WorldUpdate>,
    ui_command_tx: broadcast::Sender<UiCommand>,
    mut shutdown_rx: broadcast::Receiver<()>,
) {
    let (inner_update_tx, inner_update_rx) = watch::channel(None);
    let state = Arc::new(ServerState {
        settings,
        world_state: inner_update_rx,
        cmd_sender: ui_command_tx,
    });

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

    // Start the web server
    let mut shutdown_rx2 = shutdown_rx.resubscribe();
    let task = tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind("0.0.0.0:5555").await.unwrap();
        let shutdown_fut = async move {
            let _ = shutdown_rx2.recv().await;
        };
        tracing::info!("Webui running at http://localhost:5555");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_fut)
            .await
            .unwrap();
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
