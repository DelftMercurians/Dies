use anyhow::Result;
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use nalgebra::Vector2;
use tokio::sync::broadcast;

use super::utils::setup_vision_and_serial;

struct TestRole;

impl Role for TestRole {
    fn update(
        &mut self,
        _player_data: &dies_core::PlayerData,
        _world: &dies_core::WorldData,
    ) -> dies_executor::PlayerControlInput {
        let mut input = PlayerControlInput::new();
        input.with_position(Vector2::zeros());
        input
    }
}

pub async fn run(args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let (vision, serial) = setup_vision_and_serial(&args).await?;

    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole) as Box<dyn Role>);

    let mut builder = Executor::builder();
    builder.with_world_config(WorldConfig {
        initial_opp_goal_x: 1.0,
        is_blue: true,
    });
    builder.with_strategy(Box::new(strategy) as Box<dyn Strategy>);
    builder.with_bs_client(serial.ok_or(anyhow::anyhow!("Serial client is required"))?);
    builder.with_ssl_client(vision);
    let executor = builder.build()?;

    // Spawn webui
    let update_rx = executor.subscribe();
    let (ui_command_tx, _ui_command_rx) = broadcast::channel(16);
    let webui_shutdown_rx = stop_rx.resubscribe();
    let settings = UiSettings { can_control: true };
    let webui_task = tokio::spawn(async move {
        dies_webui::start(settings, update_rx, ui_command_tx, webui_shutdown_rx).await
    });

    executor.run_real_time(stop_rx).await?;
    webui_task.await?;

    Ok(())
}
