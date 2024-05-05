use std::time::Duration;

use anyhow::Result;
use dies_core::{PlayerData, WorldData};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
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

    let mut builder = Executor::builder();
    builder.with_world_config(WorldConfig {
        is_blue: true,
        initial_opp_goal_x: 1.0,
    });
    builder.with_strategy(Box::new(strategy) as Box<dyn Strategy>);
    builder.with_simulator(simulator, Duration::from_millis(10));

    let executor = builder.build()?;

    // Spawn webui
    let executor_rx = executor.subscribe();
    let (ui_command_tx, _) = broadcast::channel(16);
    let webui_shutdown_rx = stop_rx.resubscribe();
    tokio::spawn(
        async move { dies_webui::start(executor_rx, ui_command_tx, webui_shutdown_rx).await },
    );

    executor.run_real_time(stop_rx).await
}
