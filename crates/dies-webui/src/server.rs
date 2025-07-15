use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{Arc, RwLock},
};

use axum::{
    body::Body,
    extract::Request,
    http::header,
    middleware,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use dies_core::{
    DebugSubscriber, ExecutorInfo, ExecutorSettings, PlayerFeedbackMsg, PlayerId, ScriptError,
    TeamColor, WorldUpdate,
};
use dies_executor::{ControlMsg, ExecutorHandle, Strategy};
use tokio::sync::{broadcast, mpsc, watch};
use tower_http::services::ServeDir;
use tower_layer::Layer;

use crate::{
    executor_task::ExecutorTask, routes, ControlledTeam, ExecutorStatus, UiCommand, UiConfig,
    UiEnvironment, UiMode, UiStatus,
};

pub struct ServerState {
    pub is_live_available: bool,
    pub update_rx: watch::Receiver<Option<WorldUpdate>>,
    pub debug_sub: DebugSubscriber,
    pub basestation_feedback: RwLock<HashMap<(Option<TeamColor>, PlayerId), PlayerFeedbackMsg>>,
    pub cmd_tx: broadcast::Sender<UiCommand>,
    pub ui_mode: RwLock<UiMode>,
    pub executor_status: RwLock<ExecutorStatus>,
    pub executor_handle: RwLock<Option<ExecutorHandle>>,
    pub executor_settings: RwLock<ExecutorSettings>,
    pub script_error_tx: broadcast::Sender<ScriptError>,
    pub script_error_rx: broadcast::Receiver<ScriptError>,
    #[allow(dead_code)]
    pub robot_id_map_tx: mpsc::UnboundedSender<HashMap<PlayerId, u32>>,
    settings_file: PathBuf,
}

impl ServerState {
    pub fn new(
        is_live_available: bool,
        ui_mode: UiMode,
        settings_file: PathBuf,
        debug_sub: DebugSubscriber,
        update_rx: watch::Receiver<Option<WorldUpdate>>,
        cmd_tx: broadcast::Sender<UiCommand>,
        robot_id_map_tx: mpsc::UnboundedSender<HashMap<PlayerId, u32>>,
    ) -> Self {
        let (script_error_tx, script_error_rx) = broadcast::channel(16);

        Self {
            is_live_available,
            update_rx,
            debug_sub,
            basestation_feedback: RwLock::new(HashMap::new()),
            cmd_tx,
            ui_mode: RwLock::new(ui_mode),
            executor_status: RwLock::new(ExecutorStatus::None),
            executor_handle: RwLock::new(None),
            executor_settings: RwLock::new(ExecutorSettings::load_or_insert(&settings_file)),
            script_error_tx,
            script_error_rx,
            settings_file,
            robot_id_map_tx,
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
        let settings2 = settings.clone();
        *self.executor_settings.write().unwrap() = settings;
        let settings_file = self.settings_file.clone();
        tokio::spawn(async move { settings2.store(settings_file).await });
    }
}

/// Start the web server and executor task.
pub async fn start(config: UiConfig, shutdown_rx: broadcast::Receiver<()>, strategy: Strategy) {
    // Setup state
    let (update_tx, update_rx) = watch::channel(None);
    let (cmd_tx, cmd_rx) = broadcast::channel(16);
    let (id_map_tx, _id_map_rx) = mpsc::unbounded_channel();
    let debug_sub: DebugSubscriber = DebugSubscriber::spawn();
    let state = ServerState::new(
        config.is_live_available(),
        if config.is_live_available() {
            UiMode::Live
        } else {
            UiMode::Simulation
        },
        config.settings_file,
        debug_sub.clone(),
        update_rx.clone(),
        cmd_tx.clone(),
        id_map_tx,
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
    let state = Arc::new(state);

    // Start debug log task -- this should probably be done elsewhere, but oh well
    // let debug_log_task = {
    //     let debug_sub = debug_sub.clone();
    //     let mut shutdown_rx = shutdown_rx.resubscribe();
    //     tokio::spawn(async move {
    //         loop {
    //             tokio::select! {
    //                 data = debug_sub.wait_and_get_copy() => {
    //                     dies_logger::log_debug(&data);
    //                 }
    //                 _ = shutdown_rx.recv() => break,
    //             }
    //         }
    //     })
    // };

    // Start basestation watcher
    let basestation_task = {
        let state = Arc::clone(&state);
        let env = config.environment.clone();
        tokio::spawn(async move {
            match env {
                UiEnvironment::WithLive { mut bs_handle, .. } => {
                    while let Ok((color, msg)) = bs_handle.recv().await {
                        let mut feedback = state.basestation_feedback.write().unwrap();
                        feedback.insert((color, msg.id), msg);
                    }
                }
                UiEnvironment::SimulationOnly => {}
            }
        })
    };

    // Start executor task
    let executor_task = {
        let shutdown_rx = shutdown_rx.resubscribe();
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut executor_task =
                ExecutorTask::new(config.environment, update_tx, cmd_rx, state, strategy);
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
    basestation_task
        .await
        .expect("Shutting down basestation watcher task failed");
    web_task.await.expect("Shutting down server task failed");
    // debug_log_task
    //     .await
    //     .expect("Shutting down debug log task failed");
}

async fn start_webserver(
    port: u16,
    state: Arc<ServerState>,
    mut shutdown_rx: broadcast::Receiver<()>,
) {
    let path = std::env::current_dir()
        .unwrap()
        .join("crates")
        .join("dies-webui")
        .join("static");
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
        .route("/api/ui-mode", post(routes::post_ui_mode))
        .route("/api/command", post(routes::post_command))
        .route("/api/list", get(routes::list_files))
        .nest_service("/", serve_dir_with_csp.layer(serve_dir.clone()))
        .fallback_service(serve_dir)
        .with_state(Arc::clone(&state));

    let listener = match tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await {
        Ok(listener) => listener,
        Err(err) => {
            log::error!("Failed to bind to port {port}: {err}");
            return;
        }
    };

    let shutdown_fut = async move {
        let _ = shutdown_rx.recv().await;
    };
    log::info!("Webui running at http://localhost:{}", port);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_fut)
        .await
        .unwrap();
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
