use anyhow::{bail, Result};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, ExecutorConfig, PlayerControlInput,
};
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

pub async fn run(args: crate::Args, mut stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let (vision, serial) = setup_vision_and_serial(&args).await?;
    let serial = match serial {
        Some(serial) => serial,
        None => {
            bail!("Serial port not found");
        }
    };

    let mut strategy = AdHocStrategy::new();
    strategy.add_role(Box::new(TestRole) as Box<dyn Role>);

    let config = ExecutorConfig {
        world_config: WorldConfig {
            initial_opp_goal_x: 1.0,
            is_blue: true,
        },
    };
    let executor = Executor::new_live(config, Box::new(strategy), vision, serial);

    let mut handle = executor.handle();
    tokio::spawn(async move {
        stop_rx.recv().await.unwrap();
        handle.send(dies_executor::ControlMsg::Stop);
    });

    // Spawn webui
    // let update_rx = executor.subscribe();
    // let (ui_command_tx, _ui_command_rx) = broadcast::channel(16);
    // let webui_shutdown_rx = stop_rx.resubscribe();
    // let settings = UiSettings { can_control: true };
    // let webui_task = tokio::spawn(async move {
    //     dies_webui::start(settings, update_rx, ui_command_tx, webui_shutdown_rx).await
    // });

    executor.run_real_time().await?;
    // webui_task.await?;

    Ok(())
}
