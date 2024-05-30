use anyhow::Result;
use dies_core::SymScenario;
use dies_executor::WorldUpdate;
use dies_simulator::{Simulation, SimulationBuilder, SimulationConfig};
use dies_webui::UiSettings;
use nalgebra::{Vector2, Vector3};
use tokio::{sync::broadcast, time::Duration};

const FREQ: f64 = 60.0;
const DT: f64 = 1.0 / FREQ;

/// Run the simulation test mode.
pub async fn run(mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
    let mut simulator = single_player_simulation();

    let (update_tx, update_rx) = broadcast::channel(16);
    let (ui_command_tx, mut ui_command_rx) = broadcast::channel(16);

    // Spawn webui
    let webui_shutdown_rx = shutdown_rx.resubscribe();
    let settings = UiSettings { can_control: true };
    tokio::spawn(async move {
        dies_webui::start(settings, update_rx, ui_command_tx, webui_shutdown_rx).await
    });

    let mut is_running = false;

    // Run simulation
    let mut interval = tokio::time::interval(Duration::from_secs_f64(DT));
    loop {
        tokio::select! {
            _ = shutdown_rx.recv() => break,
            _ = interval.tick() => {
                if !is_running {
                    continue;
                }
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
                    },
                    dies_webui::UiCommand::SelectScenarioCmd { scenario } => {
                        simulator = match scenario {
                            SymScenario::Empty => empty_simulation(),
                            SymScenario::SinglePlayer => single_player_simulation(),
                            SymScenario::SinglePlayerWithoutBall => single_player_without_ball_simulation(),
                            SymScenario::TwoPlayers => two_players_simulation(),
                        };
                    },
                    dies_webui::UiCommand::StartCmd => {
                        is_running = true;
                    },
                    dies_webui::UiCommand::StopCmd => {
                        is_running = false;
                    },
                }
            }
        }
    }

    Ok(())
}

fn empty_simulation() -> Simulation {
    let simulator = SimulationBuilder::new(SimulationConfig::default()).build();
    simulator
}

fn single_player_without_ball_simulation() -> Simulation {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -500.0), 0.0)
        .build();
    simulator
}

fn single_player_simulation() -> Simulation {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -500.0), 0.0)
        .add_ball(Vector3::zeros())
        .build();
    simulator
}

fn two_players_simulation() -> Simulation {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -500.0), 0.0)
        .add_own_player(Vector2::new(0.0, 500.0), std::f64::consts::PI)
        .add_ball(Vector3::zeros())
        .build();
    simulator
}