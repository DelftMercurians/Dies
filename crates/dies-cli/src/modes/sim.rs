use std::time::Duration;

use anyhow::Result;
use dies_core::{PlayerData, WorldData};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, ExecutorConfig, PlayerControlInput,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use nalgebra::{Vector2, Vector3};
use tokio::sync::broadcast;

struct TestRole;

impl Role for TestRole {
    fn update(&mut self, _player_data: &PlayerData, _world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        input.with_position(Vector2::new(0.0, 0.0));
        input
    }
}

pub async fn run(_args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        .add_own_player(Vector2::new(0.0, -1000.0), 0.0)
        .add_ball(Vector3::zeros())
        .build();
    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole));

    let config = ExecutorConfig {
        world_config: WorldConfig {
            initial_opp_goal_x: 1.0,
            is_blue: true,
        },
    };
    let executor = Executor::new_simulation(
        config,
        Box::new(strategy),
        simulator,
        Duration::from_millis(10),
    );

    // Spawn webui
    // let executor_rx = executor.subscribe();
    // let (ui_command_tx, _) = broadcast::channel(16);
    // let webui_shutdown_rx = stop_rx.resubscribe();
    // let settings = UiSettings { can_control: true };
    // tokio::spawn(async move {
    //     println!("Webui running at http://localhost:5555");
    //     dies_webui::start(settings, executor_rx, ui_command_tx, webui_shutdown_rx).await
    // });

    executor.run_real_time().await
}
