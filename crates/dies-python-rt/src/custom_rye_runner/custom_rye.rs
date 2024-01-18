use anyhow::Result;
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    process::Command, fs,
};

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub enum CustomRyeOutput {
    None,
    Stdout,
    Log,
}

pub struct CustomRyeRunner {
    custom_rye_bin: PathBuf,
    workspace: PathBuf,
    output: CustomRyeOutput,
}

impl CustomRyeRunner {
    pub fn new(workspace: impl AsRef<Path>) -> Result<CustomRyeRunner> {
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

        let cache_dir = if let Some(mut cache_dir) = dirs::cache_dir() {
            cache_dir.push("dies");
            cache_dir.push("custom-rye");
            if !cache_dir.exists() {
                fs::create_dir_all(&cache_dir)?;
            }
            cache_dir
        }

        Ok(CustomRyeRunner {
            custom_rye_bin: cache_dir,
            workspace,
            output: CustomRyeOutput::None,
        });
    }

    pub fn get_custom_rye_bin(&self) -> PathBuf {
        self.custom_rye_bin.clone()
    }

    pub fn sync(&self) -> Result<()> {
        todo!("custom rye runner sync");
        // 1. Download python (fixed version: 3.9)
        // https://github.com/indygreg/python-build-standalone
        // 2. extract to the cache folder (code from rye)
        // 3. no virtual env in the workspace folder → create one (python -m venv)
        // 4. install dependencies from requirements.txt
    }
}
