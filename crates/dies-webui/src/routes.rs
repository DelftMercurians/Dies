use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        Json, Query, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::IntoResponse,
};
use dies_core::{DebugMap, DebugSubscriber, ScriptError, TeamColor, WorldUpdate};
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use tokio::sync::{broadcast, watch};

use crate::{
    server::ServerState, BasestationResponse, ExecutorInfoResponse, ExecutorSettingsResponse,
    GetDebugMapResponse, PostExecutorSettingsBody, PostUiCommandBody, PostUiModeBody, UiCommand,
    UiMode, UiStatus, UiWorldState, WsMessage,
};

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
    //  log::info!("Received command: {:?}", data);
    let _ = state.cmd_tx.send(data.command);
    StatusCode::OK
}

pub async fn get_executor_settings(
    state: State<Arc<ServerState>>,
) -> Json<ExecutorSettingsResponse> {
    let settings = state.executor_settings.read().unwrap().clone();
    Json(ExecutorSettingsResponse { settings })
}

pub async fn post_executor_settings(
    state: State<Arc<ServerState>>,
    Json(data): Json<PostExecutorSettingsBody>,
) -> StatusCode {
    state.update_executor_settings(data.settings);
    StatusCode::OK
}

pub async fn get_basesation_info(state: State<Arc<ServerState>>) -> Json<BasestationResponse> {
    Json(BasestationResponse {
        blue_team: state
            .basestation_feedback
            .read()
            .unwrap()
            .clone()
            .into_iter()
            .filter(|((color, _), _)| *color == Some(TeamColor::Blue))
            .map(|(_, msg)| msg)
            .collect(),
        yellow_team: state
            .basestation_feedback
            .read()
            .unwrap()
            .clone()
            .into_iter()
            .filter(|((color, _), _)| *color == Some(TeamColor::Yellow))
            .map(|(_, msg)| msg)
            .collect(),
        unknown_team: state
            .basestation_feedback
            .read()
            .unwrap()
            .clone()
            .into_iter()
            .filter(|((color, _), _)| color.is_none())
            .map(|(_, msg)| msg)
            .collect(),
    })
}

pub async fn get_debug_map(state: State<Arc<ServerState>>) -> Json<GetDebugMapResponse> {
    Json(GetDebugMapResponse {
        debug_map: state.debug_sub.get_copy(),
    })
}

pub async fn websocket(ws: WebSocketUpgrade, state: State<Arc<ServerState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| {
        handle_ws_conn(
            state.cmd_tx.clone(),
            state.update_rx.clone(),
            state.debug_sub.clone(),
            state.script_error_rx.resubscribe(),
            socket,
        )
    })
}

async fn handle_ws_conn(
    tx: broadcast::Sender<UiCommand>,
    mut world_rx: watch::Receiver<Option<WorldUpdate>>,
    debug_rx: DebugSubscriber,
    mut script_error_rx: broadcast::Receiver<ScriptError>,
    mut socket: WebSocket,
) {
    loop {
        tokio::select! {
            Some(Ok(msg)) = socket.next() => {
                handle_ws_msg(tx.clone(), msg).await
            }
            Ok(()) = world_rx.changed() => {
                if let Err(err) = handle_send_ws_world_update(&mut world_rx, &mut socket).await {
                    log::error!("Failed to send update: {}", err);
                    break;
                }
            }
            debug_map = debug_rx.wait_and_get_copy() => {
                if let Err(err) =  handle_send_debug_map_update(debug_map, &mut socket).await {
                    log::error!("Failed to send update: {}", err);
                    break;
                }
            }
            Ok(script_error) = script_error_rx.recv() => {
                if let Err(err) = handle_send_script_error(&script_error, &mut socket).await {
                    log::error!("Failed to send script error: {}", err);
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

async fn handle_send_ws_world_update(
    rx: &mut watch::Receiver<Option<WorldUpdate>>,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text_data = {
        if let Some(data) = rx.borrow_and_update().as_ref() {
            Some(serde_json::to_string(&WsMessage::WorldUpdate(
                &data.world_data,
            ))?)
        } else {
            None
        }
    };
    if let Some(text_data) = text_data {
        socket.send(Message::Text(text_data)).await?;
    }

    Ok(())
}

async fn handle_send_debug_map_update(
    debug_map: DebugMap,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text_data = serde_json::to_string(&WsMessage::Debug(&debug_map))?;
    socket.send(Message::Text(text_data)).await?;
    Ok(())
}

async fn handle_send_script_error(
    script_error: &ScriptError,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text_data = serde_json::to_string(&WsMessage::ScriptError(script_error))?;
    socket.send(Message::Text(text_data)).await?;
    Ok(())
}

#[derive(Serialize)]
pub struct FileEntry {
    pub name: String,
    pub is_dir: bool,
}

#[derive(Deserialize)]
pub struct ListQuery {
    pub dir: String,
}

pub async fn list_files(Query(query): Query<ListQuery>) -> impl IntoResponse {
    let base = PathBuf::from(".");
    let requested = PathBuf::from(&query.dir);
    let path = base.join(requested);
    if !path.exists() {
        return (StatusCode::BAD_REQUEST, "Path does not exist").into_response();
    }
    let read_dir = match fs::read_dir(&path) {
        Ok(rd) => rd,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to read directory",
            )
                .into_response()
        }
    };
    let mut entries = Vec::new();
    for entry in read_dir {
        match entry {
            Ok(e) => {
                let name = e.file_name().to_string_lossy().to_string();
                let is_dir = match e.metadata() {
                    Ok(m) => m.is_dir(),
                    Err(_) => false,
                };
                entries.push(FileEntry { name, is_dir });
            }
            Err(_) => {}
        }
    }
    axum::Json(entries).into_response()
}
