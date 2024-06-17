use axum::extract::ws::{Message, WebSocket};
use axum::extract::WebSocketUpgrade;
use axum::extract::{Json, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use dies_core::WorldUpdate;
use dies_executor::scenarios::ScenarioType;
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};

use crate::{server::ServerState, UiCommand, UiMode};

pub async fn get_world_state(state: State<Arc<ServerState>>) -> impl IntoResponse {
    let update = state.update_rx.borrow().clone();
    if let Some(update) = update {
        Json(update.world_data).into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

pub async fn get_ui_status(state: State<Arc<ServerState>>) -> impl IntoResponse {
    Json(state.ui_status()).into_response()
}

pub async fn get_scenarios() -> impl IntoResponse {
    Json(ScenarioType::get_names()).into_response()
}

pub async fn post_ui_mode(
    state: State<Arc<ServerState>>,
    Json(mode): Json<UiMode>,
) -> impl IntoResponse {
    if mode == UiMode::Live && !state.is_live_available {
        return StatusCode::BAD_REQUEST.into_response();
    }

    state.set_ui_mode(mode);
    StatusCode::OK.into_response()
}

pub async fn post_command(
    state: State<Arc<ServerState>>,
    Json(cmd): Json<UiCommand>,
) -> impl IntoResponse {
    let _ = state.cmd_tx.send(cmd);
    StatusCode::OK
}

pub async fn websocket(ws: WebSocketUpgrade, state: State<Arc<ServerState>>) -> impl IntoResponse {
    let rx = state.update_rx.clone();
    let tx = state.cmd_tx.clone();
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
                    log::error!("Failed to send update: {}", err);
                    break;
                }
            }
            else => {
                break;
            }
        }
    }

    if let Err(err) = socket.close().await {
        log::error!("Failed to close websocket: {}", err);
    }
}

async fn handle_ws_msg(tx: broadcast::Sender<UiCommand>, msg: Message) {
    match msg {
        Message::Text(text) => match serde_json::from_str::<UiCommand>(&text) {
            Ok(cmd) => {
                let _ = tx.send(cmd);
            }
            Err(err) => {
                log::error!("Failed to parse command: {}", err);
            }
        },
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
