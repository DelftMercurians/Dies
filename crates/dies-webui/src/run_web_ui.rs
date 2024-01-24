use rocket::{
    fairing::AdHoc,
    fs::{relative, FileServer},
    get, routes,
    serde::json::Json,
    Config, State,
};

use dies_core::WorldData;
use tokio::{sync::mpsc, task::JoinHandle};

use std::sync::{Arc, Mutex};

struct ServerState {
    world_data: Mutex<Option<WorldData>>,
}

pub fn spawn_webui() -> (mpsc::UnboundedSender<WorldData>, JoinHandle<()>) {
    let (tx, rx) = mpsc::unbounded_channel();

    let handle = tokio::spawn(async {
        start_rocket(rx).await;
    });

    (tx, handle)
}

#[get("/state")]
fn api(state: &State<Arc<ServerState>>) -> Json<Option<WorldData>> {
    let world_data = state.world_data.lock().unwrap();
    Json(world_data.clone())
}

async fn start_rocket(mut rx: mpsc::UnboundedReceiver<WorldData>) {
    let state = Arc::new(ServerState {
        world_data: Mutex::new(None),
    });
    let rocket = rocket::build()
        .manage(Arc::clone(&state))
        .mount("/api", routes![api])
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
            ..Default::default()
        });

    match rocket.launch().await {
        Ok(_) => log::debug!("Rocket has shut down"),
        Err(err) => {
            log::error!("Failed to start rocket: {}", err);
            panic!();
        }
    }
}
