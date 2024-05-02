use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use dies_core::{PlayerCmd, WorldData};
use std::sync::{Arc, Mutex};
use tokio::{sync::mpsc, task::JoinHandle};
use tower_http::services::ServeDir;

struct ServerState {
    world_data: Mutex<Option<WorldData>>,
    cmd_sender: mpsc::UnboundedSender<PlayerCmd>,
}

pub fn spawn_webui() -> (
    mpsc::UnboundedSender<WorldData>,
    mpsc::UnboundedReceiver<PlayerCmd>,
    JoinHandle<()>,
) {
    let (tx, rx) = mpsc::unbounded_channel();
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

    let handle = tokio::spawn(async {
        start_axum(rx, cmd_tx).await;
    });

    (tx, cmd_rx, handle)
}

async fn api(state: State<Arc<ServerState>>) -> impl IntoResponse {
    let world_data = state.world_data.lock().unwrap();
    Json(world_data.clone())
}

async fn command(state: State<Arc<ServerState>>, Json(cmd): Json<PlayerCmd>) -> impl IntoResponse {
    let _ = state.cmd_sender.send(cmd);
    StatusCode::OK
}

async fn start_axum(
    mut rx: mpsc::UnboundedReceiver<WorldData>,
    cmd_tx: mpsc::UnboundedSender<PlayerCmd>,
) {
    let state = Arc::new(ServerState {
        world_data: Mutex::new(None),
        cmd_sender: cmd_tx,
    });

    let app = Router::new()
        .route("/api/state", get(api))
        .route("/api/command", post(command))
        .nest_service("/", ServeDir::new("static"))
        .with_state(Arc::clone(&state));

    let server =
        axum::Server::bind(&"0.0.0.0:5555".parse().unwrap()).serve(app.into_make_service());

    tokio::select! {
        _ = server => {
            tracing::debug!("Axum server has shut down");
        }
        _ = async {
            // Receive world updates
            while let Some(world_data) = rx.recv().await {
                let mut state = state.world_data.lock().unwrap();
                *state = Some(world_data);
            }
        } => {
            tracing::debug!("World data sender dropped");
        }
    }
}
