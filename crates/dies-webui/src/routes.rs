use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use dies_executor::WorldUpdate;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};

use crate::server::ServerState;
use crate::server::UiCommand;
use dies_core::PlayerCmd;

pub(crate) async fn state(state: State<Arc<ServerState>>) -> impl IntoResponse {
    let update = state.world_state.borrow();
    if let Some(update) = update.as_ref() {
        Json(update.world_data.clone()).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub(crate) async fn command(
    state: State<Arc<ServerState>>,
    Json(cmd): Json<PlayerCmd>,
) -> impl IntoResponse {
    let _ = state.cmd_sender.send(UiCommand::DirectPlayerCmd { cmd });
    StatusCode::OK
}

pub(crate) async fn settings(state: State<Arc<ServerState>>) -> impl IntoResponse {
    let settings = state.settings.clone();
    Json(settings)
}

pub(crate) async fn websocket(
    ws: WebSocketUpgrade,
    state: State<Arc<ServerState>>,
) -> impl IntoResponse {
    let rx = state.world_state.clone();
    let tx = state.cmd_sender.clone();
    ws.on_upgrade(|socket| async move {
        handle_ws_conn(tx, rx, socket).await;
    })
}

async fn handle_ws_conn(
    tx: broadcast::Sender<UiCommand>,
    mut rx: watch::Receiver<Option<WorldUpdate>>,
    mut socket: WebSocket,
) {
    loop {
        tokio::select! {
            Some(Ok(msg)) = socket.next() => {
                handle_ws_msg(tx.clone(), msg).await;
            }
            Ok(()) = rx.changed() => {
                if let Err(err) = handle_send_ws_update(&mut rx, &mut socket).await {
                    tracing::error!("Failed to send update: {}", err);
                    break;
                }
            }
            else => {
                break;
            }
        }
    }

    if let Err(err) = socket.close().await {
        tracing::error!("Failed to close websocket: {}", err);
    }
}

async fn handle_ws_msg(tx: broadcast::Sender<UiCommand>, msg: Message) {
    match msg {
        Message::Text(text) => {
            if let Ok(cmd) = serde_json::from_str::<UiCommand>(&text) {
                let _ = tx.send(cmd);
            }
        }
        _ => {}
    }
}

async fn handle_send_ws_update(
    rx: &mut watch::Receiver<Option<WorldUpdate>>,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text_data = {
        if let Some(data) = rx.borrow_and_update().as_ref() {
            Some(serde_json::to_string(&data.world_data)?)
        } else {
            None
        }
    };
    if let Some(text_data) = text_data {
        socket.send(Message::Text(text_data)).await?;
    }

    Ok(())
}
