use std::{
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

use anyhow::Result;
use dies_core::PlayerId;
use dies_core::{PlayerData, WorldData};
use dies_executor::{
    strategy::{AdHocStrategy, Role, Strategy},
    Executor, PlayerControlInput,
};
use dies_simulator::{SimulationBuilder, SimulationConfig};
use dies_webui::UiSettings;
use dies_world::WorldConfig;
use nalgebra::{Vector2, Vector3};
use tokio::{sync::broadcast, time::Instant};

pub async fn run(_args: crate::Args, stop_rx: broadcast::Receiver<()>) -> Result<()> {
    let simulator = SimulationBuilder::new(SimulationConfig::default())
        // .add_own_player_with_id(GOALKEEPER_ID.as_u32(), Vector2::new(2600.0, -1000.0), 0.0)
        // .add_own_player_with_id(PASSER_ID.as_u32(), Vector2::new(-510.0, 0.0), 0.0) // Position was -1245.0, 0.0
        .add_ball(Vector3::new(-265.0, 0.0, 0.0)) // Position before was -1000.0 0.0
        .build();
    let mut strategy = AdHocStrategy::new();

    // use Arc to share the state between the passer and the receiver
    // let has_kicked_communication = Arc::new(AtomicBool::new(false));
    // strategy.add_role_with_id(
    //     GOALKEEPER_ID,
    //     Box::new(Goalkeeper {
    //         has_passer_kicked: has_kicked_communication.clone(),
    //     }),
    // );
    // let timestamp_instant = tokio::time::Instant::now();
    // strategy.add_role_with_id(
    //     PASSER_ID,
    //     Box::new(Passer {
    //         timestamp: timestamp_instant,
    //         is_armed: false,
    //         has_kicked: has_kicked_communication.clone(),
    //     }),
    // );

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
    let settings = UiSettings { can_control: true };
    tokio::spawn(async move {
        println!("Webui running at http://localhost:5555");
        dies_webui::start(settings, executor_rx, ui_command_tx, webui_shutdown_rx).await
    });

    // tokio::time::sleep(Duration::from_secs(30)).await;
    executor.run_real_time(stop_rx).await
}
