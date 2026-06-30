use std::str::FromStr;

use anyhow::{bail, Result};
use dies_logger::worker::ArrowLogger;
use log::LevelFilter;
use tokio::sync::broadcast;

use crate::cli::{Cli, StrategyMode};

pub async fn start_ui(args: Cli) -> Result<()> {
    // Set up logging
    let log_dir_path = match args.ensure_log_dir_path().await {
        Ok(path) => path,
        Err(err) => {
            bail!("Failed to create log directory: {}", err);
        }
    };
    println!("Saving logs to {}", log_dir_path.display());
    let stdout_env = env_logger::Builder::new()
        .filter_level(LevelFilter::from_str(&args.log_level).expect("Invalid log level"))
        .format_timestamp(None)
        .format_module_path(false)
        .build();
    let logger = ArrowLogger::init_with_env_logger(log_dir_path.clone(), stdout_env);
    log::set_logger(logger).unwrap(); // Safe to unwrap: no logger set yet
    log::set_max_level(log::LevelFilter::Debug);

    // Build / watch the selected strategies before the executor launches them.
    // The executor reads each binary from `target/debug`; in watch mode it also
    // hot-swaps the process whenever the binary is rebuilt. With the per-team
    // `--blue-strategy` / `--yellow-strategy` overrides there can be more than one
    // (e.g. a v0-vs-concerto benchmark), so build each distinct binary once.
    let mut selected: Vec<String> = Vec::new();
    if args.strategy != "none" {
        selected.push(args.strategy.clone());
    }
    selected.extend(args.blue_strategy.clone());
    selected.extend(args.yellow_strategy.clone());
    selected.sort();
    selected.dedup();
    // Build the explicitly-selected strategies (fatal on failure — you asked for
    // them) and watch them for hot-reload.
    for name in &selected {
        match args.strategy_mode() {
            StrategyMode::Launch => {}
            StrategyMode::Build => crate::strategy::build_strategy(name)?,
            StrategyMode::Watch => {
                crate::strategy::build_strategy(name)?;
                crate::strategy::spawn_watcher(name.clone());
            }
        }
    }

    // Build everything else the in-UI strategy picker can assign, so any choice is
    // runnable without a CLI restart: the skill-test `scenarios` crate (all bins)
    // plus every other strategy crate under `strategies/` (package == dir name).
    // Non-selected strategies build best-effort — one that doesn't compile is
    // skipped, not fatal, so it can't block startup. The scenarios crate is
    // watched so scenario edits hot-reload.
    if args.strategy_mode() != StrategyMode::Launch {
        if !selected.iter().any(|s| s == "scenarios") {
            crate::strategy::build_strategy("scenarios")?;
        }
        if args.strategy_mode() == StrategyMode::Watch {
            crate::strategy::spawn_watcher("scenarios".to_string());
        }
        if let Ok(entries) = std::fs::read_dir("strategies") {
            let mut others: Vec<String> = entries
                .flatten()
                .filter(|e| e.path().is_dir() && e.file_name() != "scenarios")
                // Build by package name (== binary name), which can differ from
                // the crate's dir name (e.g. `strategies/v0` → `v0-strategy`).
                .filter_map(|e| crate::strategy::cargo_package_name(&e.path().join("Cargo.toml")))
                .filter(|n| !selected.contains(n))
                .collect();
            others.sort();
            others.dedup();
            for pkg in others {
                if let Err(e) = crate::strategy::build_strategy(&pkg) {
                    log::warn!("Strategy `{pkg}` not built (won't be runnable): {e}");
                }
            }
        }
    }

    let (stop_tx, stop_rx) = broadcast::channel(1);
    let main_task = tokio::spawn(async move {
        let conf = match args.into_ui().await {
            Ok(conf) => conf,
            Err(err) => {
                log::error!("Failed to parse UI configuration: {}", err);
                return;
            }
        };
        dies_webui::start(conf, stop_rx).await;
    });

    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl-c");

    log::logger().flush();
    println!("Shutting down (timeout 30 seconds)... Press ctrl-c again to force shutdown");
    // Allow the logger to flush before shutting down
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;

    // Fool-proof timeout for shutdown. Generous enough to let the columnar
    // logger finish compacting the log to Parquet on close.
    std::thread::spawn(|| {
        std::thread::sleep(std::time::Duration::from_secs(30));
        eprintln!("Shutdown timed out");
        std::process::exit(1);
    });

    let shutdown_fut = async move {
        log::info!("Shutdown: broadcasting stop signal");
        stop_tx.send(()).expect("Failed to send stop signal");
        main_task.await.expect("Executor task failed");
        log::info!("Shutdown: main task joined, exiting cleanly");
    };
    tokio::select! {
        _ = shutdown_fut => {}
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Forced shutdown");
            std::process::exit(1);
        }
    };

    Ok(())
}
