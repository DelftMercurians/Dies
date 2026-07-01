//! Run one deterministic, faster-than-realtime, headless A-vs-B self-play match.
//!
//! Builds both strategy binaries, constructs a simulation executor with blocking
//! strategy IPC and a seeded simulator, runs [`Executor::run_headless`] on a
//! blocking thread, and emits the [`MatchResult`] as JSON. No webui, no
//! wall-clock pacing — same seed + same strategies → identical result.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use dies_core::{ExecutorSettings, FieldSnapshot, TeamColor};
use dies_executor::{Executor, HeadlessConfig, ScriptedEvent, ScriptedEventKind};
use dies_simulator::{SimulationBuilder, SimulationConfig};

#[allow(clippy::too_many_arguments)]
pub async fn self_play(
    blue_strategy: String,
    yellow_strategy: String,
    seed: u64,
    duration: f64,
    max_goals: Option<u32>,
    output: Option<PathBuf>,
    log_dir: Option<PathBuf>,
    snapshot: Option<String>,
    build: bool,
    release_strategies: bool,
    strategies_dir: Option<PathBuf>,
    blue_robots: usize,
    yellow_robots: usize,
    blue_card_at: Vec<f64>,
    yellow_card_at: Vec<f64>,
) -> Result<()> {
    let n_blue_robots = blue_robots.clamp(1, 6);
    let n_yellow_robots = yellow_robots.clamp(1, 6);

    // Build the scripted-event timeline from the card schedules.
    let mut scripted_events: Vec<ScriptedEvent> = blue_card_at
        .into_iter()
        .map(|t| ScriptedEvent {
            t_secs: t,
            kind: ScriptedEventKind::YellowCard {
                team: TeamColor::Blue,
            },
        })
        .chain(yellow_card_at.into_iter().map(|t| ScriptedEvent {
            t_secs: t,
            kind: ScriptedEventKind::YellowCard {
                team: TeamColor::Yellow,
            },
        }))
        .collect();
    scripted_events.sort_by(|a, b| a.t_secs.total_cmp(&b.t_secs));
    // Build both strategy binaries (release → target/release, else target/debug,
    // which is where the executor looks unless `strategies_dir` overrides it).
    if build {
        crate::strategy::build_strategy_profile(&blue_strategy, release_strategies)?;
        if yellow_strategy != blue_strategy {
            crate::strategy::build_strategy_profile(&yellow_strategy, release_strategies)?;
        }
    }

    // Resolve where the executor launches strategy binaries from: explicit
    // override wins, else the profile we (would have) built.
    let resolved_strategies_dir = strategies_dir.unwrap_or_else(|| {
        PathBuf::from(if release_strategies {
            "target/release"
        } else {
            "target/debug"
        })
    });

    let initial_snapshot = snapshot.as_deref().map(load_snapshot).transpose()?;

    let cfg = HeadlessConfig {
        blue_strategy: blue_strategy.clone(),
        yellow_strategy: yellow_strategy.clone(),
        seed,
        duration_secs: duration,
        max_goals,
        log_dir,
        initial_snapshot,
        scripted_events,
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
        settings.strategies_dir = Some(resolved_strategies_dir);

        let sim_config = SimulationConfig {
            seed,
            n_blue_robots,
            n_yellow_robots,
            // Imperfect-hardware realism model (sim2real gap), read from the
            // environment so a sweep needs no recompile. Off by default.
            realism: dies_simulator::RealismConfig::from_env(),
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

/// Resolve a `--snapshot` argument to a [`FieldSnapshot`]. A value that points
/// at an existing file (or ends in `.json`) is read directly; otherwise it is
/// treated as a snapshot name under `.dies-snapshots/`, matching where the Web
/// UI's snapshot store writes them.
fn load_snapshot(arg: &str) -> Result<FieldSnapshot> {
    let direct = Path::new(arg);
    let path = if direct.is_file() || arg.ends_with(".json") {
        direct.to_path_buf()
    } else {
        PathBuf::from(".dies-snapshots").join(format!("{arg}.json"))
    };
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("reading snapshot from {}", path.display()))?;
    serde_json::from_str(&contents).with_context(|| format!("parsing snapshot {}", path.display()))
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
