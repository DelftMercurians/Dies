use anyhow::Result;
use dies_executor::{
    roles::Goalkeeper,
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use nalgebra::Vector2;
use tokio::sync::broadcast;

use super::utils::setup_vision_and_serial;

pub async fn run(args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let (vision, serial) = setup_vision_and_serial(&args).await?;

    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(Goalkeeper::new()) as Box<dyn Role>);

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
