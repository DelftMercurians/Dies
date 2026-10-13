//! `dies logs …` — log vault client commands. Output is the command's result
//! (links, listings), so it goes to stdout.

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Subcommand;
use dies_logvault_client::{local_logs, pull, RemoteLog, VaultClient};

#[derive(Debug, Clone, Subcommand)]
pub enum LogsCmd {
    /// Upload logs (`.dieslog` files or log directories) and print share links.
    Push {
        #[clap(required = true)]
        paths: Vec<PathBuf>,
    },
    /// Upload every finished `.dieslog` in the log directory the vault lacks.
    Sync,
    /// Download a shared log (link or id) into `<log-directory>/vault/`.
    Pull { link: String },
    /// List logs on the vault (optionally filtered by name/strategy/team).
    Ls { query: Option<String> },
}

pub async fn logs(cmd: LogsCmd, vault_url: String, log_dir: PathBuf) -> Result<()> {
    let client = VaultClient::new(&vault_url);
    match cmd {
        LogsCmd::Push { paths } => {
            let mut failed = 0;
            for path in paths {
                match client.push(&path).await {
                    Ok(p) => print_pushed(&p.log, p.already_stored),
                    Err(e) => {
                        failed += 1;
                        eprintln!("{}: {e:#}", path.display());
                    }
                }
            }
            if failed > 0 {
                bail!("{failed} upload(s) failed");
            }
        }
        LogsCmd::Sync => {
            let files = local_logs(&log_dir)?;
            let (mut uploaded, mut present) = (0, 0);
            for path in &files {
                let p = client.push(path).await?;
                if p.already_stored {
                    present += 1;
                } else {
                    uploaded += 1;
                    print_pushed(&p.log, false);
                }
            }
            println!(
                "{uploaded} uploaded, {present} already in the vault ({})",
                client.base_url()
            );
        }
        LogsCmd::Pull { link } => {
            // A bare id resolves against the configured vault.
            let link = if link.contains("://") {
                link
            } else {
                format!("{}/l/{link}", client.base_url())
            };
            let path = pull(&link, &log_dir).await?;
            println!("{}", path.display());
        }
        LogsCmd::Ls { query } => {
            for l in client.list(query.as_deref()).await? {
                let kind = if l.is_match {
                    "match"
                } else if l.is_simulation {
                    "sim"
                } else {
                    "real"
                };
                println!(
                    "{:<40} {:<5} {:>8} {:>7.1} MB  {}",
                    l.name,
                    kind,
                    l.duration_s
                        .map(|d| format!("{:.0}s", d))
                        .unwrap_or_else(|| "—".into()),
                    l.size_stored as f64 / 1e6,
                    l.url
                );
            }
        }
    }
    Ok(())
}

fn print_pushed(log: &RemoteLog, already_stored: bool) {
    let note = if already_stored {
        "already stored".to_string()
    } else {
        format!(
            "{:.1} → {:.1} MB",
            log.size_uploaded as f64 / 1e6,
            log.size_stored as f64 / 1e6
        )
    };
    println!("{}  {}  ({note})", log.url, log.name);
}
