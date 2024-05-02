use anyhow::Result;
use dies_executor::WorldUpdate;
use dies_simulator::{SimulationBuilder, SimulationConfig};
use nalgebra::{Vector2, Vector3};
use tokio::{sync::broadcast, time::Duration};

const FREQ: f64 = 60.0;
const DT: f64 = 1.0 / FREQ;

/// Run the simulation test mode.
pub async fn run(mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
    let mut simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -500.0), 0.0)
        .add_ball(Vector3::zeros())
        .build();

    let (update_tx, update_rx) = broadcast::channel(16);
    let (ui_command_tx, mut ui_command_rx) = broadcast::channel(16);

    // Spawn webui
    let webui_shutdown_rx = shutdown_rx.resubscribe();
    tokio::spawn(
        async move { dies_webui::start(update_rx, ui_command_tx, webui_shutdown_rx).await },
    );

    // Run simulation
    let mut interval = tokio::time::interval(Duration::from_secs_f64(DT));
    loop {
        tokio::select! {
            _ = shutdown_rx.recv() => break,
            _ = interval.tick() => {
                simulator.step(DT);
                let state = WorldUpdate {
                    world_data: simulator.world_data(),
                };
                update_tx.send(state).unwrap();
            }
            Ok(cmd) = ui_command_rx.recv() => {
                match cmd {
                    dies_webui::UiCommand::DirectPlayerCmd { cmd } => {
                        simulator.execute_cmd(cmd);
                    }
                }
            }
        }
    }

    Ok(())
}
