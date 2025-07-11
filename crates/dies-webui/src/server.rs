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
use dies_executor::{ControlMsg, ExecutorHandle};
use tokio::sync::{broadcast, mpsc, watch};
use tower_http::services::ServeDir;
use tower_layer::Layer;

use crate::{
    executor_task::ExecutorTask, routes, ExecutorStatus, UiCommand, UiConfig, UiEnvironment,
    UiMode, UiStatus,
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
pub async fn start(config: UiConfig, shutdown_rx: broadcast::Receiver<()>) {
    // Setup state
    let (update_tx, update_rx) = watch::channel(None);
    let (cmd_tx, cmd_rx) = broadcast::channel(16);
    let (id_map_tx, _id_map_rx) = mpsc::unbounded_channel();
    let debug_sub: DebugSubscriber = DebugSubscriber::spawn();
    let state = Arc::new(ServerState::new(
        config.is_live_available(),
        if config.is_live_available() {
            UiMode::Live
        } else {
            UiMode::Simulation
        },
        config.settings_file,
        debug_sub.clone(),
        update_rx,
        cmd_tx.clone(),
        id_map_tx,
    ));

    // Start debug log task -- this should probably be done elsewhere, but oh well
    let debug_log_task = {
        let debug_sub = debug_sub.clone();
        let mut shutdown_rx = shutdown_rx.resubscribe();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    data = debug_sub.wait_and_get_copy() => {
                        dies_logger::log_debug(&data);
                    }
                    _ = shutdown_rx.recv() => break,
                }
            }
        })
    };

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
            let mut executor_task = ExecutorTask::new(config.environment, update_tx, cmd_rx, state);
            executor_task.run(shutdown_rx).await
        })
    };

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
    debug_log_task
        .await
        .expect("Shutting down debug log task failed");
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
