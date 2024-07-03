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
use crate::{ExecutorInfoResponse, PostUiCommandBody, PostUiModeBody, UiStatus, UiWorldState};

pub async fn get_world_state(state: State<Arc<ServerState>>) -> Json<UiWorldState> {
    let update = state.update_rx.borrow().clone();
    if let Some(update) = update {
        let state = UiWorldState::Loaded(update.world_data);
        Json(state)
    } else {
        Json(UiWorldState::None)
    }
}

pub async fn get_ui_status(state: State<Arc<ServerState>>) -> Json<UiStatus> {
    Json(state.ui_status())
}

pub async fn get_scenarios() -> Json<Vec<&'static str>> {
    Json(ScenarioType::get_names())
}

pub async fn get_executor_info(state: State<Arc<ServerState>>) -> Json<ExecutorInfoResponse> {
    let info = state.executor_info().await;
    Json(ExecutorInfoResponse { info })
}

pub async fn post_ui_mode(
    state: State<Arc<ServerState>>,
    Json(data): Json<PostUiModeBody>,
) -> StatusCode {
    if data.mode == UiMode::Live && !state.is_live_available {
        return StatusCode::BAD_REQUEST;
    }

    state.set_ui_mode(data.mode);
    StatusCode::OK
}

pub async fn post_command(
    state: State<Arc<ServerState>>,
    Json(data): Json<PostUiCommandBody>,
) -> StatusCode {
    let _ = state.cmd_tx.send(data.command);
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
