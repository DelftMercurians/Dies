use std::{env, path::PathBuf};

use anyhow::{bail, Result};

lazy_static::lazy_static! {
    static ref WORKSPACE_ROOT: PathBuf = try_find_workspace_root().expect("Could not find workspace root");
}

/// Try to find the workspace root.
///
/// The worksapce root is the directory that contains the Cargo.lock file.
pub fn try_find_workspace_root() -> Result<PathBuf> {
    let mut current_dir = env::current_dir()?;
    loop {
        if current_dir.join("Cargo.lock").exists() {
            return Ok(current_dir);
        }
        if !current_dir.pop() {
            break;
        }
    }
    bail!("Could not find workspace root");
}

/// Returns the absolute path to the workspace root. The value is cached.
///
/// The worksapce root is the directory that contains the Cargo.lock file.
///
/// # Panics
///
/// Panics if the workspace root cannot be found.
pub fn get_workspace_root() -> &'static PathBuf {
    &WORKSPACE_ROOT
}
