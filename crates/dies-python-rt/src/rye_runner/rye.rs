use anyhow::Result;
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Command,
};

use super::download::get_or_download_rye_bin;

/// Determines how to handle the output of rye commands.
#[derive(Debug, PartialEq)]
pub enum RyeOutput {
    None,
    Stdout,
    Log,
}

/// A runner for rye commands.
///
/// The runner is bound to a specific workspace root - ie. the directory that contains a
/// pyproject.toml file.
///
/// Upon initialization, the rye binary is downloaded if necessary. The location of the
/// directory where rye will be downloaded can be set using the environment variable
/// `DIES_RYE_DIR`, otherwise it defaults to the platform's approriate cache directory,
/// or failing that, to `{workspace}/.rye`.
///
/// The output of rye commands can be handled in three ways:
/// - `Output::None`: the output is discarded
/// - `Output::Stdout`: the output is printed to stdout
/// - `Output::Log`: the output is logged using the `log` crate (info level)
pub struct RyeRunner {
    rye_bin: PathBuf,
    workspace: PathBuf,
    output: RyeOutput,
}

impl RyeRunner {
    /// Initializes a new RyeRunner, with the given directory as workspace root,
    /// downloading the rye binary if necessary.
    ///
    /// # Errors
    ///
    /// Returns an error if
    ///  - the given directory does not exist or does not contain a pyproject.toml file
    ///  - the rye directory cannot be created
    ///  - the rye binary cannot be downloaded
    pub fn new(workspace: impl AsRef<Path>) -> Result<RyeRunner> {
        let workspace = workspace.as_ref().to_owned();
        if !workspace.is_dir() {
            anyhow::bail!("{} is not a directory", workspace.display());
        }
        if !workspace.join("pyproject.toml").is_file() {
            anyhow::bail!(
                "{} does not contain a pyproject.toml file",
                workspace.display()
            );
        }
        let rye_bin = get_or_download_rye_bin()?;
        Ok(RyeRunner {
            rye_bin,
            workspace,
            output: RyeOutput::None,
        })
    }

    /// Set how to handle the output of rye commands.
    pub fn output(&mut self, output: RyeOutput) -> &mut Self {
        self.output = output;
        self
    }

    /// Get the absolute path to the rye binary
    pub fn get_rye_bin(&self) -> PathBuf {
        self.rye_bin.clone()
    }

    /// Runs `rye <args>`.
    ///
    /// # Errors
    ///
    /// Returns an error if
    ///  - the command cannot be run
    ///  - the command exits with a non-zero exit code
    fn exec(&self, args: &[&str]) -> Result<()> {
        log::info!("Running rye {}", args.join(" "));
        let mut cmd = Command::new(&self.rye_bin);
        cmd.current_dir(&self.workspace)
            .args(args)
            .stdout(match self.output {
                RyeOutput::None => std::process::Stdio::null(),
                RyeOutput::Stdout => std::process::Stdio::inherit(),
                RyeOutput::Log => std::process::Stdio::piped(),
            });

        let status = if self.output == RyeOutput::Log {
            // Stream output to log
            let mut child = cmd.spawn()?;
            let stdout = child.stdout.take().unwrap();
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    log::info!("[rye {}]: {}", args.join(" "), line);
                }
            }
            child.wait()?
        } else {
            // Run command
            cmd.status()?
        };

        match status.code() {
            Some(0) => {
                log::info!("rye exited with code 0");
                Ok(())
            }
            Some(code) => anyhow::bail!("rye exited with code {}", code),
            None => anyhow::bail!("rye terminated by signal"),
        }
    }

    /// Runs `rye sync`.
    ///
    /// # Errors
    ///
    /// Returns an error if
    /// - the command cannot be run
    /// - the command exits with a non-zero exit code
    pub fn sync(&self) -> Result<()> {
        self.exec(&["sync"])
    }
}
