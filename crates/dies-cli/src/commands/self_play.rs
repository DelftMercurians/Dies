//! Run one deterministic, faster-than-realtime, headless A-vs-B self-play match.
//!
//! Builds both strategy binaries, constructs a simulation executor with blocking
//! strategy IPC and a seeded simulator, runs [`Executor::run_headless`] on a
//! blocking thread, and emits the [`MatchResult`] as JSON. No webui, no
//! wall-clock pacing — same seed + same strategies → identical result.

use std::path::PathBuf;

use anyhow::{Context, Result};
use dies_core::ExecutorSettings;
use dies_executor::{Executor, HeadlessConfig};
use dies_simulator::{SimulationBuilder, SimulationConfig};

#[allow(clippy::too_many_arguments)]
pub async fn self_play(
    blue_strategy: String,
    yellow_strategy: String,
    seed: u64,
    duration: f64,
    max_goals: Option<u32>,
    output: Option<PathBuf>,
    build: bool,
) -> Result<()> {
    // Build both strategy binaries into target/debug (where the executor looks).
    if build {
        crate::strategy::build_strategy(&blue_strategy)?;
        if yellow_strategy != blue_strategy {
            crate::strategy::build_strategy(&yellow_strategy)?;
        }
    }

    let cfg = HeadlessConfig {
        blue_strategy: blue_strategy.clone(),
        yellow_strategy: yellow_strategy.clone(),
        seed,
        duration_secs: duration,
        max_goals,
    };

    // run_headless is a blocking sync loop (blocking IPC + free-running sim), so
    // run it off the async runtime.
    let result = tokio::task::spawn_blocking(move || -> Result<_> {
        let mut settings = ExecutorSettings::load_or_insert(&PathBuf::from("dies-settings.json"));
        settings.team_configuration.blue_active = true;
        settings.team_configuration.yellow_active = true;
        settings.team_configuration.blue_strategy = Some(blue_strategy);
        settings.team_configuration.yellow_strategy = Some(yellow_strategy);
        settings.strategy_blocking = true;

        let sim_config = SimulationConfig {
            seed,
            ..Default::default()
        };
        let simulator = SimulationBuilder::default_seeded(sim_config)
            .with_controlled_teams(true, true)
            .build();

        let executor = Executor::new_simulation(settings, simulator);
        executor.run_headless(cfg)
    })
    .await
    .context("headless match thread panicked")??;

    let json = serde_json::to_string_pretty(&result)?;
    match output {
        Some(path) => {
            std::fs::write(&path, &json)
                .with_context(|| format!("writing match result to {}", path.display()))?;
            eprintln!("Match finished: {}", summary(&result));
            eprintln!("Wrote result to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn summary(r: &dies_executor::MatchResult) -> String {
    format!(
        "{} {} - {} {} (seed {}, {:.1}s sim, {:?})",
        r.blue_strategy,
        r.blue_score,
        r.yellow_score,
        r.yellow_strategy,
        r.seed,
        r.duration_secs,
        r.end_reason,
    )
}
