//! Strategy build & watch orchestration.
//!
//! Replaces the old `just dev` / `just webdev` build steps. The CLI is
//! responsible for turning strategy *sources* into a *binary*; the executor's
//! strategy host is responsible for turning that binary into a *process* (and,
//! in `--strategy-mode watch`, hot-swapping it when the binary changes).

use std::path::Path;
use std::process::Command;
use std::sync::mpsc;
use std::time::Duration;

use anyhow::{bail, Result};
use notify::{RecursiveMode, Watcher};

/// Directories whose changes should trigger a strategy rebuild in watch mode:
/// the strategies themselves plus the strategy-facing crate closure. We watch
/// all of `strategies/` recursively rather than a single crate so the binary
/// name (e.g. `concerto`) need not match its directory (`strategies/concerto`);
/// `cargo build -p <name>` is a no-op when an unrelated change comes in.
const WATCH_DIRS: &[&str] = &[
    "strategies",
    "crates/dies-strategy-api/src",
    "crates/dies-strategy-protocol/src",
    "crates/dies-strategy-runner/src",
    "crates/dies-core/src",
];

/// Extract the `[package] name` from a Cargo.toml with a minimal hand parse (no
/// toml dependency). The strategy binary name equals the package name — which is
/// what `cargo build -p <name>` and the executor's `strategies_dir/<name>` use —
/// and a crate's dir name can differ from its package name.
pub fn cargo_package_name(cargo_toml: &Path) -> Option<String> {
    let text = std::fs::read_to_string(cargo_toml).ok()?;
    let mut in_package = false;
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_package = t == "[package]";
            continue;
        }
        if in_package {
            if let Some(rest) = t.strip_prefix("name") {
                if let Some(val) = rest.trim_start().strip_prefix('=') {
                    let name = val.trim().trim_matches('"');
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Build a strategy binary with `cargo build -p <name>` (debug profile, to match
/// the `target/debug` directory the executor launches from). Returns an error if
/// cargo cannot be spawned or the build fails.
pub fn build_strategy(name: &str) -> Result<()> {
    build_strategy_profile(name, false)
}

/// Build a strategy binary, optionally in release profile. Release strategies
/// land in `target/release` (run the executor with a matching `strategies_dir`);
/// debug strategies in `target/debug` (the executor's default). Used by the
/// self-play harness to trade compile time for faster matches.
pub fn build_strategy_profile(name: &str, release: bool) -> Result<()> {
    let profile = if release { "release" } else { "debug" };
    println!("Building strategy `{name}` ({profile})...");
    // Scenarios live as separate `[[bin]]` targets inside the single `scenarios`
    // package, so `cargo build -p <scenario>` would fail (no such package). Detect
    // a scenario bin by its source file and build it as `-p scenarios --bin <name>`.
    // Plain strategies are their own package, so `-p <name>` is correct.
    let is_scenario = Path::new("strategies/scenarios/src/bin")
        .join(format!("{name}.rs"))
        .exists();
    let mut args = vec!["build"];
    if is_scenario {
        args.extend(["-p", "scenarios", "--bin", name]);
    } else {
        args.extend(["-p", name]);
    }
    if release {
        args.push("--release");
    }
    match Command::new("cargo").args(&args).status() {
        Ok(status) if status.success() => {
            println!("Strategy `{name}` built ({profile})");
            Ok(())
        }
        Ok(status) => bail!(
            "`cargo build {} ` ({profile}) failed with {status}",
            args.join(" ")
        ),
        Err(e) => bail!("failed to run `cargo build` for `{name}` ({profile}): {e}"),
    }
}

/// Spawn a background thread that watches strategy sources and rebuilds `name`
/// on every change. The running executor picks up the rebuilt binary via its own
/// mtime watch, so no further signalling is needed here.
pub fn spawn_watcher(name: String) {
    std::thread::Builder::new()
        .name("strategy-watcher".into())
        .spawn(move || {
            if let Err(e) = watch_loop(&name) {
                log::error!("Strategy watcher stopped: {e}");
            }
        })
        .expect("failed to spawn strategy watcher thread");
}

fn watch_loop(name: &str) -> Result<()> {
    let (tx, rx) = mpsc::channel();
    let mut watcher = notify::recommended_watcher(move |res| {
        let _ = tx.send(res);
    })?;

    for dir in WATCH_DIRS {
        let path = Path::new(dir);
        if path.exists() {
            watcher.watch(path, RecursiveMode::Recursive)?;
        } else {
            log::warn!("Watch directory not found, skipping: {dir}");
        }
    }
    log::info!("Watching strategy sources for changes (hot-reload)");

    loop {
        // Block until the first change of a burst.
        match rx.recv() {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => {
                log::warn!("Strategy watch error: {e}");
                continue;
            }
            Err(_) => break, // sender dropped — watcher gone
        }
        // Debounce: drain the rest of the burst until things go quiet.
        while rx.recv_timeout(Duration::from_millis(300)).is_ok() {}

        log::info!("Change detected, rebuilding strategy `{name}`");
        if let Err(e) = build_strategy(name) {
            log::warn!("Rebuild failed (fix and save again): {e}");
        }
    }

    Ok(())
}
