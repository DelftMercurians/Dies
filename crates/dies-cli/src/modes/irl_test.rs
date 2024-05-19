use anyhow::Result;
use dies_executor::Executor;
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use tokio::sync::broadcast;

use super::utils::setup_vision_and_serial;

pub async fn run(args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let (vision, serial) = setup_vision_and_serial(&args).await?;
    let (ui_command_tx, mut ui_command_rx) = broadcast::channel(16);

    // Setup direct command channel
    let (direct_cmd_tx, direct_cmd_rx) = broadcast::channel(16);
    let mut stop_rx2 = stop_rx.resubscribe();
    let dc_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = stop_rx2.recv() => break,
                Ok(cmd) = ui_command_rx.recv() => {
                    match cmd {
                        dies_webui::UiCommand::DirectPlayerCmd { cmd } => {
                            if let Err(err) = direct_cmd_tx.send(cmd) {
                                tracing::error!("Failed to send direct command: {}", err);
                            }
                        }
                    }
                }
                else => break,
            }
        }
    });

    let mut builder = Executor::builder();
    builder.with_direct_control(direct_cmd_rx);
    builder.with_world_config(WorldConfig {
        initial_opp_goal_x: 1.0,
        is_blue: true,
    });
    builder.with_bs_client(serial.ok_or(anyhow::anyhow!("Serial client is required"))?);
    builder.with_ssl_client(vision);
    let executor = builder.build()?;

    // Spawn webui
    let update_rx = executor.subscribe();
    let webui_shutdown_rx = stop_rx.resubscribe();
    let settings = UiSettings { can_control: true };
    let webui_task = tokio::spawn(async move {
        dies_webui::start(settings, update_rx, ui_command_tx, webui_shutdown_rx).await
    });

    executor.run_real_time(stop_rx).await?;

    // Wait for all tasks to finish
    tokio::try_join!(dc_task, webui_task)
        .map_err(|err| anyhow::anyhow!("Failed to join tasks: {}", err))?;

    Ok(())
}
