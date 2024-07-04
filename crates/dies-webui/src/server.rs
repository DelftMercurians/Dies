use axum::{
    routing::{get, post},
    Router,
};
use dies_core::{ControllerSettings, ExecutorInfo, WorldUpdate};
use dies_executor::{ControlMsg, ExecutorHandle};
use std::sync::{Arc, RwLock};
use tokio::sync::{broadcast, watch};
use tower_http::services::ServeDir;

use crate::{
    executor_task::ExecutorTask, routes, ExecutorStatus, UiCommand, UiConfig, UiMode, UiStatus,
};

pub struct ServerState {
    pub is_live_available: bool,
    pub update_rx: watch::Receiver<Option<WorldUpdate>>,
    pub cmd_tx: broadcast::Sender<UiCommand>,
    pub ui_mode: RwLock<UiMode>,
    pub executor_status: RwLock<ExecutorStatus>,
    pub executor_handle: RwLock<Option<ExecutorHandle>>,
    pub controller_settings: RwLock<ControllerSettings>,
}

impl ServerState {
    pub fn new(
        is_live_available: bool,
        update_rx: watch::Receiver<Option<WorldUpdate>>,
        cmd_tx: broadcast::Sender<UiCommand>,
    ) -> Self {
        Self {
            is_live_available,
            update_rx,
            cmd_tx,
            ui_mode: RwLock::new(UiMode::Simulation),
            executor_status: RwLock::new(ExecutorStatus::None),
            executor_handle: RwLock::new(None),
            controller_settings: RwLock::new(ControllerSettings::default()),
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
            self.executor_handle
                .read()
                .unwrap()
                .as_ref()
                .map(|h| h.info())
        };
        if let Some(rx) = rx {
            rx.recv().await
        } else {
            None
        }
    }

    /// Update the controller settings.
    pub fn update_controller_settings(&self, settings: ControllerSettings) {
        if let Some(handle) = self.executor_handle.read().unwrap().as_ref() {
            handle.send(ControlMsg::UpdateControllerSettings(settings.clone()));
        }
        *self.controller_settings.write().unwrap() = settings;
    }
}

/// Start the web server and executor task.
pub async fn start(config: UiConfig, shutdown_rx: broadcast::Receiver<()>) {
    // Setup state
    let (update_tx, update_rx) = watch::channel(None);
    let (cmd_tx, cmd_rx) = broadcast::channel(16);
    let state = Arc::new(ServerState::new(
        config.is_live_available(),
        update_rx,
        cmd_tx,
    ));

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
    web_task.await.expect("Shutting down server task failed");
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
    let app = Router::new()
        .route("/api/executor", get(routes::get_executor_info))
        .route("/api/world-state", get(routes::get_world_state))
        .route("/api/ui-status", get(routes::get_ui_status))
        .route("/api/scenarios", get(routes::get_scenarios))
        .route("/api/ws", get(routes::websocket))
        .route("/api/controller", get(routes::get_controller_settings))
        .route("/api/controller", post(routes::post_controller_settings))
        .route("/api/ui-mode", post(routes::post_ui_mode))
        .route("/api/command", post(routes::post_command))
        .nest_service("/", serve_dir.clone())
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
