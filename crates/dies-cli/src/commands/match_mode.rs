//! Match mode — a lightweight, robust orchestrator on top of the normal dies
//! run, for use during real matches.
//!
//! Two phases:
//!  1. **Checklist** (once, before launch): pick the controlled team color, force
//!     safe pre-match settings (clear the vision field mask, turn `ignore_gc`
//!     off), then walk an operator-confirmed markdown checklist.
//!  2. **Supervisor**: build concerto once, then loop launching the whole dies
//!     binary as a child process and **relaunching it on any exit** (crash,
//!     panic, segfault). Ctrl-C kills the child and exits without relaunching.
//!
//! The child is started with `--strategy-param warmup=true`, so concerto poses in
//! the logo formation at startup; the operator disables `warmup` from the Web UI
//! when ready to play (nothing disables it automatically — the default checklist
//! warns about this).

use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};

use std::io::Write as _;

use anyhow::{Context, Result};
use dies_core::{ExecutorSettings, FieldMask};
use tokio::{
    io::{AsyncBufReadExt, BufReader},
    process::Command,
};

use crate::cli::Cli;

/// Backoff before relaunching the child after an unexpected exit.
const RELAUNCH_BACKOFF: Duration = Duration::from_secs(2);

pub async fn run(cli: Cli, checklist: PathBuf) -> Result<()> {
    let mut stdin = BufReader::new(tokio::io::stdin());

    // --- Phase 1a: pick the controlled team color ---------------------------
    let team = prompt_team(&mut stdin).await?;

    // --- Phase 1b: force safe pre-match settings into the settings file -----
    // The child loads this file on startup, so the safe state survives every
    // relaunch.
    force_safe_settings(&cli.settings_file)?;

    // --- Phase 1c: walk the operator checklist ------------------------------
    run_checklist(&mut stdin, &checklist).await?;

    // --- Phase 2a: build concerto once so children can `--launch` -----------
    println!("\nBuilding concerto...");
    crate::strategy::build_strategy("concerto").context("Failed to build concerto")?;

    // --- Phase 2b: supervisor loop ------------------------------------------
    println!(
        "\nStarting match mode (team: {team}). The Web UI will be at \
         http://localhost:{}.\nPress Ctrl-C to stop.\n",
        cli.webui_port
    );
    supervise(&cli, &team).await
}

/// Prompt for the controlled team color, re-asking on invalid input.
async fn prompt_team(stdin: &mut BufReader<tokio::io::Stdin>) -> Result<String> {
    loop {
        print!("Which team are we controlling? [blue/yellow/both] (blue): ");
        std::io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.read_line(&mut line).await? == 0 {
            anyhow::bail!("stdin closed before a team was selected");
        }
        match line.trim().to_lowercase().as_str() {
            "" | "blue" | "b" => return Ok("blue".to_owned()),
            "yellow" | "y" => return Ok("yellow".to_owned()),
            "both" => return Ok("both".to_owned()),
            other => println!("  '{other}' is not a valid team — pick blue, yellow, or both."),
        }
    }
}

/// Force the two dangerous leftover test states off: clear the vision field mask
/// (full field) and turn `ignore_gc` off for both teams. Rewrites the settings
/// file in place.
fn force_safe_settings(settings_file: &PathBuf) -> Result<()> {
    let mut settings = ExecutorSettings::load_or_insert(settings_file);
    settings.tracker_settings.field_mask = FieldMask::default();
    settings.blue_team_settings.ignore_gc = false;
    settings.yellow_team_settings.ignore_gc = false;
    let json =
        serde_json::to_string_pretty(&settings).context("Failed to serialize executor settings")?;
    std::fs::write(settings_file, json)
        .with_context(|| format!("Failed to write {}", settings_file.display()))?;
    println!("✓ field mask cleared (full field)");
    println!("✓ GC ignore off (both teams)");
    Ok(())
}

/// Read the markdown checklist and confirm each item with the operator. Lines
/// that look like checklist items (`- [ ]` / `- [x]` / `- `) are prompted one at
/// a time (press Enter to confirm); other lines (headings, blanks) are printed
/// for context. A missing file is a warning, not a hard failure.
async fn run_checklist(stdin: &mut BufReader<tokio::io::Stdin>, path: &PathBuf) -> Result<()> {
    let contents = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => {
            println!(
                "\n⚠ Checklist file '{}' not found — skipping checklist.",
                path.display()
            );
            return Ok(());
        }
    };

    println!("\n=== Pre-match checklist ===");
    for raw in contents.lines() {
        let line = raw.trim_end();
        match checklist_item(line) {
            Some(item) if !item.is_empty() => {
                print!("  [ ] {item}  (Enter to confirm) ");
                std::io::stdout().flush().ok();
                let mut buf = String::new();
                if stdin.read_line(&mut buf).await? == 0 {
                    anyhow::bail!("stdin closed during checklist");
                }
                println!("  [x] {item}");
            }
            _ => {
                if !line.is_empty() {
                    println!("{line}");
                }
            }
        }
    }
    println!("=== Checklist complete ===");
    Ok(())
}

/// If `line` is a markdown checklist item, return its text (without the marker).
fn checklist_item(line: &str) -> Option<&str> {
    let t = line.trim_start();
    for prefix in ["- [ ] ", "- [x] ", "- [X] ", "- ", "* "] {
        if let Some(rest) = t.strip_prefix(prefix) {
            return Some(rest.trim());
        }
    }
    None
}

/// Supervise a child dies process: (re)launch it, restarting on any exit until
/// the operator presses Ctrl-C.
async fn supervise(cli: &Cli, team: &str) -> Result<()> {
    // SIGINT is delivered to the whole foreground process group, so the child
    // gets it too and shuts down gracefully; we just need to avoid relaunching.
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = shutdown.clone();
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                shutdown.store(true, Ordering::SeqCst);
            }
        });
    }

    let exe = std::env::current_exe().context("Failed to locate the dies executable")?;
    let mut launches: u64 = 0;
    loop {
        launches += 1;
        if launches > 1 {
            println!("\n[match supervisor] relaunch #{launches}...");
        }
        let mut child = build_child_command(&exe, cli, team)
            .spawn()
            .context("Failed to spawn dies child process")?;

        tokio::select! {
            biased;
            _ = tokio::signal::ctrl_c() => {
                println!("\n[match supervisor] Ctrl-C — stopping, no relaunch.");
                let _ = child.kill().await;
                break;
            }
            status = child.wait() => {
                if shutdown.load(Ordering::SeqCst) {
                    break;
                }
                match status {
                    Ok(s) => eprintln!(
                        "[match supervisor] dies exited unexpectedly ({s}); restarting in {}s.",
                        RELAUNCH_BACKOFF.as_secs()
                    ),
                    Err(e) => eprintln!(
                        "[match supervisor] failed to wait on dies ({e}); restarting in {}s.",
                        RELAUNCH_BACKOFF.as_secs()
                    ),
                }
                tokio::time::sleep(RELAUNCH_BACKOFF).await;
            }
        }
    }
    Ok(())
}

/// Reconstruct the child `dies` invocation: forward the operator's run flags and
/// add the match-mode overrides.
fn build_child_command(exe: &std::path::Path, cli: &Cli, team: &str) -> Command {
    let mut cmd = Command::new(exe);
    cmd.arg("--settings-file")
        .arg(&cli.settings_file)
        .args(["--webui-port", &cli.webui_port.to_string()])
        .args(["--ui-mode", &cli.ui_mode])
        .args(["--serial-port", &cli.serial_port.to_arg()])
        .args(["--connection-mode", cli.connection_mode.to_arg()])
        .args(["--vision-addr", &cli.vision_addr.to_string()])
        .args(["--gc-addr", &cli.gc_addr.to_string()])
        .args(["--vision-delay-ms", &cli.vision_delay_ms.to_string()])
        .args(["--log-level", &cli.log_level])
        .args(["--log-directory", &cli.log_directory]);
    if let Some(ref interface) = cli.interface {
        cmd.args(["--interface", interface]);
    }
    // Match-mode overrides.
    cmd.arg("--auto-start")
        .arg("--is-match")
        .args(["--controlled-teams", team])
        .args(["--strategy", "concerto"])
        .args(["--strategy-param", "warmup=true"])
        .arg("--launch");
    // Inherit stdio so the child's logs and the Web UI banner show in this
    // terminal.
    cmd
}
