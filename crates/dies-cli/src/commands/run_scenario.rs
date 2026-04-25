//! Run a JS test scenario headlessly in simulation or live.
//!
//! Boots a regular executor, immediately sends a `StartScenario` message, streams
//! log entries to stdout, and waits for the scenario to finish (or abort).

use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use dies_basestation_client::{BasestationClientConfig, BasestationHandle};
use dies_core::{ExecutorSettings, TeamColor};
use dies_executor::{ControlMsg, Executor, ExecutorHandle};
use dies_simulator::SimulationBuilder;
use dies_ssl_client::VisionClient;
use dies_test_driver::TestStatus;
use dies_webui::UiMode;
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::oneshot;

pub async fn run_scenario(
    path: PathBuf,
    team: TeamColor,
    stream_logs: bool,
    mode: UiMode,
    bs_config: Option<BasestationClientConfig>,
    ssl_config: Option<dies_ssl_client::SslClientConfig>,
    timeout_secs: Option<f64>,
) -> Result<()> {
    if !path.exists() {
        anyhow::bail!("scenario not found: {}", path.display());
    }

    let (handle_tx, handle_rx) = oneshot::channel::<ExecutorHandle>();
    let scenario_path = path.clone();
    let thread = std::thread::Builder::new()
        .name("dies-executor".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("build executor runtime");
            rt.block_on(async move {
                let settings = ExecutorSettings::default();
                let executor = match mode {
                    UiMode::Live => {
                        let bs_client =
                            BasestationHandle::spawn(bs_config.expect("Bs config is required"))
                                .expect("Failed to start BS client");
                        let vision_client =
                            VisionClient::new(ssl_config.expect("SSL config is required"))
                                .await
                                .expect("Failed to start vision client");
                        Executor::new_live(settings, vision_client, bs_client)
                    }
                    UiMode::Simulation => {
                        let simulator = SimulationBuilder::default().build();
                        Executor::new_simulation(settings, simulator)
                    }
                };
                let handle = executor.handle();
                handle.send(ControlMsg::StartScenario {
                    path: scenario_path,
                    team: Some(team),
                });
                let _ = handle_tx.send(handle);
                if let Err(err) = executor.run_real_time().await {
                    eprintln!("executor error: {:?}", err);
                }
            });
        })?;

    let handle = match handle_rx.await {
        Ok(h) => h,
        Err(_) => anyhow::bail!("executor thread failed to send handle"),
    };

    if stream_logs {
        let mut log_rx = handle.log_bus.subscribe();
        tokio::spawn(async move {
            loop {
                match log_rx.recv().await {
                    Ok(entry) => {
                        let tag = entry.tag.as_deref().unwrap_or("");
                        match entry.value_json {
                            Some(v) => println!(
                                "[{:?}]{}{} {} {}",
                                entry.level,
                                if tag.is_empty() { "" } else { " " },
                                tag,
                                entry.message,
                                v
                            ),
                            None => println!(
                                "[{:?}]{}{} {}",
                                entry.level,
                                if tag.is_empty() { "" } else { " " },
                                tag,
                                entry.message
                            ),
                        }
                    }
                    Err(RecvError::Lagged(_)) => continue,
                    Err(RecvError::Closed) => break,
                }
            }
        });
    }

    // Watch scenario status — terminate executor once the scenario finishes.
    let status_handle = handle.clone();
    let mut status_rx = handle.scenario_status_rx.clone();
    tokio::spawn(async move {
        // Wait until the status leaves Idle/Starting (i.e. actually starts running),
        // then wait for it to leave Running again.
        let mut saw_running = false;
        loop {
            if status_rx.changed().await.is_err() {
                return;
            }
            let s = status_rx.borrow().clone();
            match s {
                TestStatus::Running { .. } => saw_running = true,
                TestStatus::Completed { .. }
                | TestStatus::Failed { .. }
                | TestStatus::Aborted => {
                    if saw_running {
                        // Give pending log lines a moment to flush, then stop.
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        status_handle.send(ControlMsg::Stop);
                        return;
                    }
                }
                _ => {}
            }
        }
    });

    // Ctrl-C handling.
    let stop_handle = handle.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        println!("\n(Ctrl-C) stopping executor");
        stop_handle.send(ControlMsg::Stop);
    });

    // Wait for the executor thread (bounded by timeout).
    let dur = timeout_secs
        .map(Duration::from_secs_f64)
        .unwrap_or(Duration::from_secs(3600));

    let join_handle = tokio::task::spawn_blocking(move || thread.join());
    match tokio::time::timeout(dur, join_handle).await {
        Ok(Ok(Ok(()))) => {}
        Ok(Ok(Err(e))) => eprintln!("executor thread panicked: {:?}", e),
        Ok(Err(e)) => eprintln!("join error: {:?}", e),
        Err(_) => {
            eprintln!("scenario timed out; forcing stop");
            handle.send(ControlMsg::Stop);
        }
    }
    Ok(())
}
