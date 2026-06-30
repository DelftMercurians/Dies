use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        Json, Path, Query, State, WebSocketUpgrade,
    },
    http::StatusCode,
    response::IntoResponse,
};
use dies_core::{DebugMap, DebugSubscriber, TeamColor, WorldUpdate};
use futures::StreamExt;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::{broadcast, watch};
use tokio::time::MissedTickBehavior;

/// Max rate at which the debug map is pushed to a connected client.
///
/// The debug map is rebuilt many times per executor tick (every `debug_*`
/// call notifies subscribers), so sending on every notification floods the
/// browser faster than it can parse + re-render, and the WS backlog grows
/// without bound. We coalesce to the latest snapshot at a fixed rate instead.
const DEBUG_SEND_HZ: u64 = 30;

use crate::{
    server::ServerState, BasestationResponse, ConsoleLogMessage, ExecutorInfoResponse,
    ExecutorSettingsResponse, FieldSnapshot, GetDebugMapResponse, LogInfo, LogsResponse,
    PostExecutorSettingsBody, PostUiCommandBody, PostUiModeBody, ReplayState, SaveSnapshotBody,
    SettingsSnapshot, SettingsSnapshotsResponse, SnapshotsResponse, StrategiesResponse, UiCommand,
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

/// Baseline + auto-captured history for the settings explore/revert UI.
pub async fn get_settings_snapshots(
    state: State<Arc<ServerState>>,
) -> Json<SettingsSnapshotsResponse> {
    Json(state.settings_snapshots())
}

/// Mark the current live config as the known-good baseline.
pub async fn post_settings_baseline(state: State<Arc<ServerState>>) -> Json<SettingsSnapshot> {
    Json(state.set_settings_baseline())
}

/// List saved simulator field-state snapshots.
pub async fn get_snapshots(state: State<Arc<ServerState>>) -> Json<SnapshotsResponse> {
    Json(SnapshotsResponse {
        names: state.list_field_snapshots(),
    })
}

/// Fetch a single field-state snapshot by name.
pub async fn get_snapshot(
    state: State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> Result<Json<FieldSnapshot>, StatusCode> {
    state
        .get_field_snapshot(&name)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// Save a field-state snapshot under a name (overwrites an existing one).
pub async fn post_snapshot(
    state: State<Arc<ServerState>>,
    Json(body): Json<SaveSnapshotBody>,
) -> StatusCode {
    if state.save_field_snapshot(&body.name, &body.snapshot) {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

/// Delete a field-state snapshot by name.
pub async fn delete_snapshot(
    state: State<Arc<ServerState>>,
    Path(name): Path<String>,
) -> StatusCode {
    if state.delete_field_snapshot(&name) {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    }
}

/// Extract the `[package] name` from a Cargo.toml with a minimal hand parse (no
/// toml dependency). The strategy binary name equals the package name, which is
/// what `cargo build -p <name>` and the executor's `strategies_dir/<name>` both
/// use — and a crate's dir name can differ from its package name.
fn cargo_package_name(cargo_toml: &std::path::Path) -> Option<String> {
    let text = fs::read_to_string(cargo_toml).ok()?;
    let mut in_package = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = t.strip_prefix("name") {
                if let Some(val) = rest.trim_start().strip_prefix('=') {
                    let name = val.trim().trim_matches('"');
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

/// List the binaries the strategy picker can assign to a team: full strategy
/// crates under `strategies/` (each subdir except `scenarios`, by package name)
/// and the skill-test binaries in `strategies/scenarios/src/bin/`. Read from
/// source so the list is complete regardless of what's currently built; the CLI
/// builds them all at startup so they're runnable. Both kinds are plain
/// `Strategy` binaries.
pub async fn get_strategies() -> Json<StrategiesResponse> {
    let mut strategies = Vec::new();
    if let Ok(entries) = fs::read_dir("strategies") {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() || entry.file_name() == "scenarios" {
                continue;
            }
            if let Some(name) = cargo_package_name(&path.join("Cargo.toml")) {
                strategies.push(name);
            }
        }
    }
    strategies.sort();

    let mut scenarios = Vec::new();
    if let Ok(entries) = fs::read_dir("strategies/scenarios/src/bin") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                scenarios.push(stem.to_string());
            }
        }
    }
    scenarios.sort();

    Json(StrategiesResponse {
        strategies,
        scenarios,
    })
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
        base_info: state.base_info.read().unwrap().clone(),
    })
}

pub async fn get_debug_map(state: State<Arc<ServerState>>) -> Json<GetDebugMapResponse> {
    Json(GetDebugMapResponse {
        debug_map: state.debug_sub.get_copy(),
    })
}

/// List recorded logs (directories and `.dieslog` zips) under the log dir, with
/// metadata read cheaply from each `meta.json`. When a log directory and its
/// sibling zip both exist, only the directory is listed (faster to load).
pub async fn get_logs(state: State<Arc<ServerState>>) -> Json<LogsResponse> {
    let mut logs = Vec::new();
    if let Ok(entries) = fs::read_dir(&state.log_directory) {
        let entries: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        let dir_names: std::collections::HashSet<String> = entries
            .iter()
            .filter(|p| p.is_dir())
            .filter_map(|p| p.file_name().and_then(|n| n.to_str()).map(String::from))
            .collect();

        for path in &entries {
            let is_zip = path.extension().map(|e| e == "dieslog").unwrap_or(false);
            let is_dir = path.is_dir();
            if !is_zip && !is_dir {
                continue;
            }
            // Skip a zip if its sibling directory is also present.
            if is_zip {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if dir_names.contains(stem) {
                        continue;
                    }
                }
            }
            let meta = match dies_logger::MetaJson::read_any(path) {
                Ok(m) => m,
                Err(_) => continue, // not a log dir / unreadable
            };
            let duration_s = meta.duration_s();
            let end_unix = duration_s.map(|d| meta.session_start_unix + d);
            logs.push(LogInfo {
                name: path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string(),
                path: path.to_string_lossy().to_string(),
                session_start_unix: meta.session_start_unix,
                duration_s,
                end_unix,
                frame_count: meta.frame_count,
                is_simulation: meta.is_simulation,
                blue_strategy: meta.blue_strategy,
                yellow_strategy: meta.yellow_strategy,
                is_zip,
            });
        }
    }
    // Newest first.
    logs.sort_by(|a, b| {
        b.session_start_unix
            .partial_cmp(&a.session_start_unix)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Json(LogsResponse { logs })
}

pub async fn websocket(ws: WebSocketUpgrade, state: State<Arc<ServerState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| {
        handle_ws_conn(
            state.cmd_tx.clone(),
            state.update_rx.clone(),
            state.debug_sub.clone(),
            state.console_log_tx.subscribe(),
            state.replay_state.subscribe(),
            socket,
        )
    })
}

#[allow(clippy::too_many_arguments)]
async fn handle_ws_conn(
    tx: broadcast::Sender<UiCommand>,
    mut world_rx: watch::Receiver<Option<WorldUpdate>>,
    debug_rx: DebugSubscriber,
    mut console_rx: broadcast::Receiver<ConsoleLogMessage>,
    mut replay_rx: watch::Receiver<ReplayState>,
    mut socket: WebSocket,
) {
    // Rate-limit debug pushes: tick at a fixed rate and send the latest
    // snapshot, coalescing the many per-tick debug updates into one message.
    let mut debug_interval = tokio::time::interval(Duration::from_millis(1000 / DEBUG_SEND_HZ));
    debug_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

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
            _ = debug_interval.tick() => {
                let debug_map = debug_rx.get_copy();
                if let Err(err) =  handle_send_debug_map_update(debug_map, &mut socket).await {
                    log::error!("Failed to send update: {}", err);
                    break;
                }
            }
            console_entry = console_rx.recv() => {
                match console_entry {
                    Ok(entry) => {
                        if let Err(err) = handle_send_console_log(&entry, &mut socket).await {
                            log::error!("Failed to send console log: {}", err);
                            break;
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(broadcast::error::RecvError::Closed) => {}
                }
            }
            Ok(()) = replay_rx.changed() => {
                let replay = replay_rx.borrow_and_update().clone();
                let text = serde_json::to_string(&WsMessage::ReplayState(&replay));
                if let Ok(text) = text {
                    if socket.send(Message::Text(text)).await.is_err() {
                        break;
                    }
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
    if let Message::Text(text) = msg {
        match serde_json::from_str::<UiCommand>(&text) {
            Ok(cmd) => {
                let _ = tx.send(cmd);
            }
            Err(err) => {
                log::error!("Failed to parse command: {}", err);
            }
        }
    }
}

async fn handle_send_ws_world_update(
    rx: &mut watch::Receiver<Option<WorldUpdate>>,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text_data = {
        if let Some(data) = rx.borrow_and_update().as_ref() {
            Some(serde_json::to_string(&WsMessage::WorldUpdate(data))?)
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

async fn handle_send_console_log(
    entry: &ConsoleLogMessage,
    socket: &mut WebSocket,
) -> anyhow::Result<()> {
    let text = serde_json::to_string(&WsMessage::ConsoleLog(entry))?;
    socket.send(Message::Text(text)).await?;
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
    for e in read_dir.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        let is_dir = match e.metadata() {
            Ok(m) => m.is_dir(),
            Err(_) => false,
        };
        entries.push(FileEntry { name, is_dir });
    }
    axum::Json(entries).into_response()
}
