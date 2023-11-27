use anyhow::Result;
use crossbeam::{channel::unbounded, select};
use dies_core::{
    EnvEvent, EnvReceiver, EnvSender, RuntimeEvent, RuntimeMsg, RuntimeReceiver, RuntimeSender,
};
use dies_world::{WorldConfig, WorldTracker};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

enum Update {
    EnvEvent(EnvEvent),
    RuntimeEvent(RuntimeEvent),
}

pub fn run(
    (env_tx, mut env_rx): (Box<dyn EnvSender>, Box<dyn EnvReceiver>),
    (mut rt_tx, mut rt_rx): (Box<dyn RuntimeSender>, Box<dyn RuntimeReceiver>),
    should_stop: Arc<AtomicBool>,
) -> Result<()> {
    let (env_ev_tx, env_ev_rx) = unbounded::<EnvEvent>();
    let (rt_ev_tx, rt_ev_rx) = unbounded::<RuntimeEvent>();

    // Launch the receiver threads
    let should_stop_env_rx = Arc::clone(&should_stop);
    let env_rx_thread = std::thread::spawn(move || {
        while !should_stop_env_rx.load(Ordering::Relaxed) {
            match env_rx.recv() {
                Ok(ev) => {
                    if let Err(err) = env_ev_tx.send(ev) {
                        log::error!("Failed to send env event: {}", err);
                    }
                }
                Err(err) => {
                    log::error!("Failed to receive env event: {}", err);
                }
            }
        }
    });
    let should_stop_rt_rx = Arc::clone(&should_stop);
    let rt_rx_thread = std::thread::spawn(move || {
        while !should_stop_rt_rx.load(Ordering::Relaxed) {
            match rt_rx.recv() {
                Ok(ev) => {
                    if let Err(err) = rt_ev_tx.send(ev) {
                        log::error!("Failed to send rt event: {}", err);
                    }
                }
                Err(err) => {
                    log::error!("Failed to receive rt event: {}", err);
                }
            }
        }
    });

    // Create world tracker
    // TODO: Make this configurable
    let mut tracker = WorldTracker::new(WorldConfig {
        is_blue: true,
        initial_opp_goal_x: 1.0,
    });

    // Main loop
    // 1. Receive events from the environment
    // 2. Process events with the world tracker
    // 3. Send updates to the runtime
    // 4. Send commands to the environment
    while !should_stop.load(Ordering::Relaxed) {
        let update = select! {
            recv(env_ev_rx) -> msg => Update::EnvEvent(msg?),
            recv(rt_ev_rx) -> msg => Update::RuntimeEvent(msg?),
        };

        match update {
            Update::EnvEvent(EnvEvent::VisionMsg(vision_msg)) => {
                // Process event with world tracker
                tracker.update_from_protobuf(&vision_msg);

                // Send world data to runtime
                if let Some(world_data) = tracker.get() {
                    if let Err(err) = rt_tx.send(&RuntimeMsg::World(world_data)) {
                        log::error!("Failed to send world data to runtime: {}", err);
                    }
                }
            }
            Update::EnvEvent(EnvEvent::GcRefereeMsg(_)) => {}
            Update::RuntimeEvent(ev) => match ev {
                RuntimeEvent::PlayerCmd(cmd) => {
                    // Send command to environment
                    if let Err(err) = env_tx.send_player(cmd) {
                        log::error!("Failed to send player cmd to env: {}", err);
                    }
                }
                RuntimeEvent::Debug { msg } => {
                    log::debug!("Runtime debug: {}", msg);
                }
                RuntimeEvent::Crash { msg } => {
                    log::error!("Runtime crash: {}", msg);
                    break;
                }
            },
        }
    }

    // Stop threads
    should_stop.store(true, Ordering::Relaxed);
    env_rx_thread.join().unwrap();
    rt_rx_thread.join().unwrap();

    Ok(())
}
