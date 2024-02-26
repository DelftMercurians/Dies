use anyhow::{bail, Context, Result};
use std::{env, path::PathBuf};

lazy_static::lazy_static! {
    static ref WORKSPACE_ROOT: PathBuf = try_find_workspace_root().expect("Could not find workspace root");
}

/// Try to find the workspace root.
///
/// The worksapce root is the directory that contains a `py` directory and a `pyproject.toml` file.
pub fn try_find_workspace_root() -> Result<PathBuf> {
    let mut current_dir = env::current_dir().context("Failed to get the current directory from the env variable")?;
    loop {
        let py_dir = current_dir.join("py");
        let pyproject_toml = current_dir.join("pyproject.toml");
        if py_dir.is_dir() && pyproject_toml.is_file() {
            return Ok(current_dir.canonicalize().context("Failed to canonicalize the current directory path")?);
        }
        if !current_dir.pop() {
            break;
        }
    }
    bail!("Could not find workspace root");
}

/// Returns the absolute path to the workspace root. The value is cached.
///
/// The worksapce root is the directory that contains a `py` directory and a `pyproject.toml` file.
///
/// # Panics
///
/// Panics if the workspace root cannot be found.
pub fn get_workspace_root() -> &'static PathBuf {
    &WORKSPACE_ROOT
}
