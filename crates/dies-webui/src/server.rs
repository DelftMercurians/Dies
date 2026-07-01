use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    body::Body,
    extract::Request,
    http::header,
    middleware,
    response::IntoResponse,
    routing::{delete, get, post},
    Router,
};
use dies_core::{
    BaseStationInfo, DebugSubscriber, ExecutorInfo, ExecutorSettings, PlayerFeedbackMsg, PlayerId,
    TeamColor, WorldUpdate,
};
use dies_executor::{ControlMsg, ExecutorHandle};
use tokio::sync::{broadcast, watch};
use tower_http::services::ServeDir;
use tower_layer::Layer;

use crate::{
    executor_task::ExecutorTask, routes, settings_store::SettingsStore, snapshots::SnapshotStore,
    ConsoleLogMessage, ControlledTeam, ExecutorStatus, FieldSnapshot, ReplayState,
    SettingsSnapshot, SettingsSnapshotsResponse, UiCommand, UiConfig, UiEnvironment, UiMode,
    UiStatus,
};

pub struct ServerState {
    pub is_live_available: bool,
    pub update_rx: watch::Receiver<Option<WorldUpdate>>,
    pub debug_sub: DebugSubscriber,
    pub basestation_feedback: RwLock<HashMap<(Option<TeamColor>, PlayerId), PlayerFeedbackMsg>>,
    /// Latest RF basestation info (radios, version, channel), for the test bench.
    pub base_info: RwLock<Option<BaseStationInfo>>,
    pub cmd_tx: broadcast::Sender<UiCommand>,
    pub ui_mode: RwLock<UiMode>,
    pub executor_status: RwLock<ExecutorStatus>,
    pub executor_handle: RwLock<Option<ExecutorHandle>>,
    pub executor_settings: RwLock<ExecutorSettings>,
    /// Stable broadcast of backend log lines, fed by the global logger's console
    /// observer, consumed by WS clients for the console panel.
    pub console_log_tx: broadcast::Sender<ConsoleLogMessage>,
    /// Latest replay-player state, pushed to WS clients on change.
    pub replay_state: watch::Sender<ReplayState>,
    /// Directory where session logs live (for the replay browser).
    pub log_directory: PathBuf,
    settings_file: PathBuf,
    /// Local store backing the settings baseline + auto-history (explore/revert).
    settings_store: SettingsStore,
    /// Local store of saved simulator field-state snapshots.
    snapshot_store: SnapshotStore,
}

impl ServerState {
    pub fn new(
        is_live_available: bool,
        ui_mode: UiMode,
        settings_file: PathBuf,
        log_directory: PathBuf,
        debug_sub: DebugSubscriber,
        update_rx: watch::Receiver<Option<WorldUpdate>>,
        cmd_tx: broadcast::Sender<UiCommand>,
    ) -> Self {
        let settings = ExecutorSettings::load_or_insert(&settings_file);
        // Snapshots live in a gitignored `.dies-settings/` next to the settings
        // file, so the explore/revert history stays machine-local.
        let settings_parent = settings_file
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let settings_store = SettingsStore::load(settings_parent.join(".dies-settings"));
        let snapshot_store = SnapshotStore::load(settings_parent.join(".dies-snapshots"));
        // Backend logs are chatty; a lagging WS client drops old entries (handled
        // in the WS loop) rather than blocking the logging thread.
        let (console_log_tx, _) = broadcast::channel::<ConsoleLogMessage>(1024);
        {
            let tx = console_log_tx.clone();
            dies_logger::set_console_observer(move |record| {
                let ts_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as i64;
                let _ = tx.send(ConsoleLogMessage {
                    level: record.level().into(),
                    target: record.target().to_string(),
                    message: format!("{}", record.args()),
                    ts_ms,
                });
            });
        }
        let (replay_state, _) = watch::channel(ReplayState::default());
        Self {
            is_live_available,
            update_rx,
            debug_sub,
            basestation_feedback: RwLock::new(HashMap::new()),
            base_info: RwLock::new(None),
            cmd_tx,
            ui_mode: RwLock::new(ui_mode),
            executor_status: RwLock::new(ExecutorStatus::None),
            executor_handle: RwLock::new(None),
            executor_settings: RwLock::new(settings),
            console_log_tx,
            replay_state,
            log_directory,
            settings_file,
            settings_store,
            snapshot_store,
        }
    }

    pub fn set_ui_mode(&self, new_mode: UiMode) {
        *self.ui_mode.write().unwrap() = new_mode;
    }

    pub fn set_executor_status(&self, info: ExecutorStatus) {
        *self.executor_status.write().unwrap() = info;
    }

    pub fn ui_status(&self) -> UiStatus {
        UiStatus {
            is_live_available: self.is_live_available,
            ui_mode: *self.ui_mode.read().unwrap(),
            executor: self.executor_status.read().unwrap().clone(),
        }
    }

    /// Get the current executor info.
    pub async fn executor_info(&self) -> Option<ExecutorInfo> {
        let rx = {
            match self
                .executor_handle
                .read()
                .unwrap()
                .as_ref()
                .map(|h| h.info())
            {
                Some(Ok(rx)) => Some(rx),
                Some(Err(_)) => {
                    // If the receiver is gone, remove the handle
                    // self.executor_handle.write().unwrap().take();
                    None
                }
                None => None,
            }
        };
        if let Some(rx) = rx {
            rx.recv().await
        } else {
            None
        }
    }

    /// Update the controller settings.
    pub fn update_executor_settings(&self, settings: ExecutorSettings) {
        if let Some(handle) = self.executor_handle.read().unwrap().as_ref() {
            handle.send(ControlMsg::UpdateSettings(settings.clone()));
        }
        // Record into the (debounced) explore/revert history.
        self.settings_store.note_edit(settings.clone());
        let settings2 = settings.clone();
        *self.executor_settings.write().unwrap() = settings;
        let settings_file = self.settings_file.clone();
        tokio::spawn(async move { settings2.store(settings_file).await });
    }

    /// Mark the current live config as the known-good baseline.
    pub fn set_settings_baseline(&self) -> SettingsSnapshot {
        let current = self.executor_settings.read().unwrap().clone();
        self.settings_store.set_baseline(current)
    }

    /// Current baseline + auto-history for the explore/revert UI.
    pub fn settings_snapshots(&self) -> SettingsSnapshotsResponse {
        self.settings_store.snapshots()
    }

    pub fn list_field_snapshots(&self) -> Vec<String> {
        self.snapshot_store.list()
    }

    pub fn get_field_snapshot(&self, name: &str) -> Option<FieldSnapshot> {
        self.snapshot_store.get(name)
    }

    pub fn save_field_snapshot(&self, name: &str, snapshot: &FieldSnapshot) -> bool {
        self.snapshot_store.save(name, snapshot)
    }

    pub fn delete_field_snapshot(&self, name: &str) -> bool {
        self.snapshot_store.delete(name)
    }
}

/// Start the web server and executor task.
pub async fn start(config: UiConfig, shutdown_rx: broadcast::Receiver<()>) {
    // Setup state
    let (update_tx, update_rx) = watch::channel(None);
    let (cmd_tx, cmd_rx) = broadcast::channel(16);
    let debug_sub: DebugSubscriber = DebugSubscriber::instance();
    let state = ServerState::new(
        config.is_live_available(),
        if config.is_live_available() {
            UiMode::Live
        } else {
            UiMode::Simulation
        },
        config.settings_file,
        config.log_directory.clone(),
        debug_sub.clone(),
        update_rx.clone(),
        cmd_tx.clone(),
    );
    state
        .executor_settings
        .write()
        .unwrap()
        .team_configuration
        .blue_active = matches!(
        config.controlled_teams,
        ControlledTeam::Blue | ControlledTeam::Both
    );
    state
        .executor_settings
        .write()
        .unwrap()
        .team_configuration
        .yellow_active = matches!(
        config.controlled_teams,
        ControlledTeam::Yellow | ControlledTeam::Both
    );
    // Set IPC strategy for active teams
    if let Some(ref strategy) = config.strategy {
        let mut settings = state.executor_settings.write().unwrap();
        if settings.team_configuration.blue_active {
            settings.team_configuration.blue_strategy = Some(strategy.clone());
        }
        if settings.team_configuration.yellow_active {
            settings.team_configuration.yellow_strategy = Some(strategy.clone());
        }
    }
    // Per-team strategy overrides (e.g. an asymmetric benchmark). Each one both
    // selects that team's binary and activates the team, independent of
    // `--controlled-teams` / `--strategy`.
    if let Some(ref strategy) = config.blue_strategy {
        let mut settings = state.executor_settings.write().unwrap();
        settings.team_configuration.blue_active = true;
        settings.team_configuration.blue_strategy = Some(strategy.clone());
    }
    if let Some(ref strategy) = config.yellow_strategy {
        let mut settings = state.executor_settings.write().unwrap();
        settings.team_configuration.yellow_active = true;
        settings.team_configuration.yellow_strategy = Some(strategy.clone());
    }
    // Propagate the dev-only hot-reload flag into the executor settings.
    state.executor_settings.write().unwrap().hot_reload = config.hot_reload;
    state.executor_settings.write().unwrap().vision_delay_ms = config.vision_delay_ms;
    // Seed CLI-provided initial strategy params (match mode's `warmup=true`).
    state
        .executor_settings
        .write()
        .unwrap()
        .initial_strategy_params = config.initial_strategy_params.clone();
    // Mark real-match runs (match subcommand) for meta.json + the MATCH START marker.
    state.executor_settings.write().unwrap().is_match = config.is_match;
    let state = Arc::new(state);

    // Start basestation watcher + test-bench task (telemetry cache, base info,
    // and direct 50 Hz robot command streaming).
    let basestation_task = {
        let state = Arc::clone(&state);
        let env = config.environment.clone();
        let cmd_rx = cmd_tx.subscribe();
        let shutdown_rx = shutdown_rx.resubscribe();
        tokio::spawn(async move {
            crate::bench::run(state, env, cmd_rx, shutdown_rx).await;
        })
    };

    // Start executor task
    let executor_task = {
        let shutdown_rx = shutdown_rx.resubscribe();
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut executor_task = ExecutorTask::new(config.environment, update_tx, cmd_rx, state);
            if config.auto_start {
                log::info!("Starting executor automatically in 2 seconds...");
                tokio::spawn(async move {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    log::info!("Starting executor automatically");
                    cmd_tx.send(UiCommand::Start).unwrap();
                });
            }
            executor_task.run(shutdown_rx).await;
        })
    };

    if config.calibration_mode {
        let mut update_rx = state.update_rx.clone();
        tokio::spawn(async move {
            let file =
                std::fs::File::create("calibration_data.csv").expect("Failed to create CSV file");
            let mut writer = csv::Writer::from_writer(file);

            // Build CSV header
            let mut header = vec![
                "timestamp_received".to_owned(),
                "timestamp_capture".to_owned(),
                "dt".to_owned(),
                "game_state".to_owned(),
                "operating_team".to_owned(),
                "ball_detected".to_owned(),
                "ball_pos_x".to_owned(),
                "ball_pos_y".to_owned(),
                "ball_pos_z".to_owned(),
                "ball_vel_x".to_owned(),
                "ball_vel_y".to_owned(),
                "ball_vel_z".to_owned(),
                "blue_players_count".to_owned(),
                "yellow_players_count".to_owned(),
            ];

            // Add player headers for each team (assuming max 6 players per team)
            for team in ["blue", "yellow"] {
                for player_id in 0..6 {
                    header.extend_from_slice(&[
                        format!("{team}_player_{player_id}_id"),
                        format!("{team}_player_{player_id}_pos_x"),
                        format!("{team}_player_{player_id}_pos_y"),
                        format!("{team}_player_{player_id}_vel_x"),
                        format!("{team}_player_{player_id}_vel_y"),
                        format!("{team}_player_{player_id}_yaw"),
                        format!("{team}_player_{player_id}_raw_pos_x"),
                        format!("{team}_player_{player_id}_raw_pos_y"),
                        format!("{team}_player_{player_id}_raw_yaw"),
                        format!("{team}_player_{player_id}_angular_speed"),
                        format!("{team}_player_{player_id}_breakbeam_detected"),
                    ]);
                }
            }

            writer
                .write_record(&header)
                .expect("Failed to write CSV header");

            while let Ok(()) = update_rx.changed().await {
                let update = update_rx.borrow_and_update().clone();
                if let Some(update) = update {
                    let world_data = &update.world_data;

                    // Prepare ball data
                    let (
                        ball_detected,
                        ball_pos_x,
                        ball_pos_y,
                        ball_pos_z,
                        ball_vel_x,
                        ball_vel_y,
                        ball_vel_z,
                    ) = if let Some(ball) = &world_data.ball {
                        (
                            ball.detected,
                            ball.position.x,
                            ball.position.y,
                            ball.position.z,
                            ball.velocity.x,
                            ball.velocity.y,
                            ball.velocity.z,
                        )
                    } else {
                        (false, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    };

                    // Start building the record
                    let mut record = vec![
                        world_data.t_received.to_string(),
                        world_data.t_capture.to_string(),
                        world_data.dt.to_string(),
                        world_data.game_state.game_state.to_string(),
                        format!("{:?}", world_data.game_state.operating_team),
                        ball_detected.to_string(),
                        ball_pos_x.to_string(),
                        ball_pos_y.to_string(),
                        ball_pos_z.to_string(),
                        ball_vel_x.to_string(),
                        ball_vel_y.to_string(),
                        ball_vel_z.to_string(),
                        world_data.blue_team.len().to_string(),
                        world_data.yellow_team.len().to_string(),
                    ];

                    // Add blue team player data
                    for player_id in 0..6 {
                        if let Some(player) = world_data
                            .blue_team
                            .iter()
                            .find(|p| p.id.as_u32() == player_id)
                        {
                            record.extend_from_slice(&[
                                player.id.as_u32().to_string(),
                                player.position.x.to_string(),
                                player.position.y.to_string(),
                                player.velocity.x.to_string(),
                                player.velocity.y.to_string(),
                                player.yaw.radians().to_string(),
                                player.raw_position.x.to_string(),
                                player.raw_position.y.to_string(),
                                player.raw_yaw.radians().to_string(),
                                player.angular_speed.to_string(),
                                player.breakbeam_ball_detected.to_string(),
                            ]);
                        } else {
                            // Fill with zeros for missing players
                            record.extend_from_slice(&[
                                player_id.to_string(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "false".into(),
                            ]);
                        }
                    }

                    // Add yellow team player data
                    for player_id in 0..6 {
                        if let Some(player) = world_data
                            .yellow_team
                            .iter()
                            .find(|p| p.id.as_u32() == player_id)
                        {
                            record.extend_from_slice(&[
                                player.id.as_u32().to_string(),
                                player.position.x.to_string(),
                                player.position.y.to_string(),
                                player.velocity.x.to_string(),
                                player.velocity.y.to_string(),
                                player.yaw.radians().to_string(),
                                player.raw_position.x.to_string(),
                                player.raw_position.y.to_string(),
                                player.raw_yaw.radians().to_string(),
                                player.angular_speed.to_string(),
                                player.breakbeam_ball_detected.to_string(),
                            ]);
                        } else {
                            // Fill with zeros for missing players
                            record.extend_from_slice(&[
                                player_id.to_string(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "0.0".into(),
                                "false".into(),
                            ]);
                        }
                    }

                    writer
                        .write_record(&record)
                        .expect("Failed to write CSV record");
                    writer.flush().expect("Failed to flush CSV writer");
                }
            }
        });
    }

    // Start the web server
    let web_task =
        tokio::spawn(async move { start_webserver(config.port, state, shutdown_rx).await });

    // Graceful shutdown
    executor_task
        .await
        .expect("Shutting down executor task failed");
    log::info!("Shutdown: executor task joined; joining basestation task");
    basestation_task
        .await
        .expect("Shutting down basestation watcher task failed");
    log::info!("Shutdown: basestation task joined; joining web task");
    web_task.await.expect("Shutting down server task failed");
    log::info!("Shutdown: web task joined; dies_webui::start returning");
}

async fn start_webserver(
    port: u16,
    state: Arc<ServerState>,
    mut shutdown_rx: broadcast::Receiver<()>,
) {
    // Resolve relative to this crate's source dir (baked at compile time) rather
    // than the process cwd, so the binary serves the bundle regardless of where
    // it's launched from. The bundle is produced by build.rs for release builds.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("static");
    let serve_dir = ServeDir::new(path);
    let serve_dir_with_csp = middleware::from_fn(add_csp_header);

    let app = Router::new()
        .route("/api/executor", get(routes::get_executor_info))
        .route("/api/world-state", get(routes::get_world_state))
        .route("/api/ui-status", get(routes::get_ui_status))
        .route("/api/ws", get(routes::websocket))
        .route("/api/settings", get(routes::get_executor_settings))
        .route("/api/basestation", get(routes::get_basesation_info))
        .route("/api/debug", get(routes::get_debug_map))
        .route("/api/settings", post(routes::post_executor_settings))
        .route(
            "/api/settings/snapshots",
            get(routes::get_settings_snapshots),
        )
        .route(
            "/api/settings/baseline",
            post(routes::post_settings_baseline),
        )
        .route("/api/ui-mode", post(routes::post_ui_mode))
        .route("/api/command", post(routes::post_command))
        .route("/api/list", get(routes::list_files))
        .route("/api/logs", get(routes::get_logs))
        .route("/api/strategies", get(routes::get_strategies))
        .route(
            "/api/snapshots",
            get(routes::get_snapshots).post(routes::post_snapshot),
        )
        .route(
            "/api/snapshots/:name",
            get(routes::get_snapshot).delete(routes::delete_snapshot),
        )
        .nest_service("/", serve_dir_with_csp.layer(serve_dir.clone()))
        .fallback_service(serve_dir)
        .with_state(Arc::clone(&state));

    // Prefer a dual-stack ([::]) listener so Firefox — which resolves `localhost`
    // to IPv6 `::1` first — doesn't hit a refused connection and fall back to
    // IPv4 (a per-request stall). Fall back to IPv4-only if IPv6 is unavailable.
    let listener = match bind_dual_stack(port) {
        Ok(listener) => listener,
        Err(err) => {
            log::warn!(
                "Dual-stack bind on [::]:{port} failed ({err}); falling back to 0.0.0.0:{port}"
            );
            match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
                Ok(listener) => listener,
                Err(err) => {
                    log::error!("Failed to bind to port {port}: {err}");
                    return;
                }
            }
        }
    };

    let shutdown_fut = async move {
        let _ = shutdown_rx.recv().await;
        log::info!("Shutdown: web server got stop signal, starting graceful shutdown (waiting for open connections, e.g. WebSockets)");
    };
    log::info!("Webui running at http://localhost:{}", port);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_fut)
        .await
        .unwrap();
    log::info!("Shutdown: axum::serve graceful shutdown complete");
}

/// Bind a dual-stack TCP listener on `[::]:port` accepting both IPv6 and
/// IPv4-mapped connections (`IPV6_V6ONLY=false`), returned as a tokio listener.
fn bind_dual_stack(port: u16) -> std::io::Result<tokio::net::TcpListener> {
    use socket2::{Domain, Protocol, Socket, Type};
    use std::net::{Ipv6Addr, SocketAddr};

    let addr: SocketAddr = (Ipv6Addr::UNSPECIFIED, port).into();
    let socket = Socket::new(Domain::IPV6, Type::STREAM, Some(Protocol::TCP))?;
    socket.set_only_v6(false)?;
    socket.set_reuse_address(true)?;
    socket.set_nonblocking(true)?;
    socket.bind(&addr.into())?;
    socket.listen(1024)?;
    tokio::net::TcpListener::from_std(socket.into())
}

async fn add_csp_header(req: Request<Body>, next: axum::middleware::Next) -> impl IntoResponse {
    let mut response = next.run(req).await;
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        "script-src 'self' 'unsafe-eval' 'unsafe-inline'"
            .parse()
            .unwrap(),
    );
    response
}
