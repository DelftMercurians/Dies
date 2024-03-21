use rocket::{
    fairing::AdHoc,
    fs::{relative, FileServer},
    get, post, routes,
    serde::json::Json,
    Config, State,
};

use dies_core::{PlayerCmd, WorldData};
use tokio::{sync::mpsc, task::JoinHandle};

use std::sync::{Arc, Mutex};

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
        start_rocket(rx, cmd_tx).await;
    });

    (tx, cmd_rx, handle)
}

#[get("/state")]
fn api(state: &State<Arc<ServerState>>) -> Json<Option<WorldData>> {
    let world_data = state.world_data.lock().unwrap();
    Json(world_data.clone())
}

#[post("/command", data = "<cmd>")]
fn command(state: &State<Arc<ServerState>>, cmd: Json<PlayerCmd>) {
    let cmd = cmd.into_inner();
    let _ = state.cmd_sender.send(cmd);
}

async fn start_rocket(
    mut rx: mpsc::UnboundedReceiver<WorldData>,
    cmd_tx: mpsc::UnboundedSender<PlayerCmd>,
) {
    let state = Arc::new(ServerState {
        world_data: Mutex::new(None),
        cmd_sender: cmd_tx,
    });
    let rocket = rocket::build()
        .manage(Arc::clone(&state))
        .mount("/api", routes![api, command])
        .mount("/", FileServer::from(relative!("static")))
        .attach(AdHoc::on_liftoff("on_start", |rocket| {
            Box::pin(async move {
                let shutdown = rocket.shutdown();
                tokio::spawn(async move {
                    // Receive world updates
                    while let Some(world_data) = rx.recv().await {
                        let mut state = state.world_data.lock().unwrap();
                        *state = Some(world_data);
                    }

                    // Sender dropped, we should stop
                    shutdown.notify();
                });
            })
        }))
        .configure(Config {
            port: 5555,
            log_level: rocket::config::LogLevel::Off,
            ..Default::default()
        });

    match rocket.launch().await {
        Ok(_) => tracing::debug!("Rocket has shut down"),
        Err(err) => {
            tracing::error!("Failed to start rocket: {}", err);
            panic!();
        }
    }
}
