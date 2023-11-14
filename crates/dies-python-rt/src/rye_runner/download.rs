use anyhow::Result;
use const_format::formatcp;
use dies_core::workspace_utils::get_workspace_root;
use flate2::read::GzDecoder;
use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter},
    path::PathBuf,
};

const VERSION: &str = "0.15.2";

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
const BINARY: &str = "rye-x86_64-macos.gz";

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const BINARY: &str = "rye-x86_64-linux.gz";

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const BINARY: &str = "rye-aarch64-macos.gz";

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
const BINARY: &str = "rye-aarch64-linux.gz";

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
const BINARY: &str = "rye-x86_64-windows.exe";

#[cfg(all(target_os = "windows", target_arch = "aarch64"))]
const BINARY: &str = "rye-aarch64-windows.exe";

const DOWNLOAD_URL: &str = formatcp!(
    "https://github.com/mitsuhiko/rye/releases/download/{}/{}",
    VERSION,
    BINARY
);

/// Returns the absolute path to the rye binary, downloading it if necessary.
///
/// The location of the directory where rye will be downloaded can be set using the
/// environment variable `DIES_RYE_DIR`, otherwise it defaults to the platform's
/// approriate cache directory, or failing that, to `{workspace}/.rye`.
///
/// # Errors
///
/// Returns an error if the rye binary cannot be downloaded or if the .rye directory
/// cannot be created.
pub fn get_or_download_rye_bin() -> Result<PathBuf> {
    let rye_dir = get_rye_dir()?;
    let path = if cfg!(windows) {
        rye_dir.join("rye.exe")
    } else {
        rye_dir.join("rye")
    };

    if path.exists() {
        log::info!("Found rye binary at {}", path.display());
        return Ok(path);
    }

    log::info!("Downloading rye binary to {}", path.display());
    let response = reqwest::blocking::get(DOWNLOAD_URL)?;
    let bytes = response.bytes()?;
    let file = File::create(path.clone())?;
    let mut decoder = GzDecoder::new(&bytes[..]);
    let mut writer = BufWriter::new(&file);
    io::copy(&mut decoder, &mut writer)?;

    set_permission(&file)?;

    log::info!("Downloaded rye binary to {}", path.display());

    Ok(path)
}

/// Returns the absolute path to the `.rye` directory, creating it if it does not exist.
///
/// # Errors
///
/// Returns an error if the `.rye` directory cannot be created, eg. due to permissions.
fn get_rye_dir() -> Result<PathBuf> {
    // Check the environment variable first
    if let Ok(dir_from_env) = env::var("DIES_RYE_DIR") {
        if let Ok(path) = PathBuf::from(dir_from_env).canonicalize() {
            if !path.exists() {
                fs::create_dir_all(&path)?;
            }
            return Ok(path);
        }
    }

    // If not set in environment, check the platform's appropriate cache directory
    if let Some(mut cache_dir) = dirs::cache_dir() {
        cache_dir.push("dies");
        cache_dir.push("rye");
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }
        return Ok(cache_dir);
    }

    // If all else fails, use {workspace}/.rye
    let workspace_root = get_workspace_root();
    let path = workspace_root.join(".rye");
    if !path.exists() {
        fs::create_dir_all(&path)?;
    }
    Ok(path)
}

#[cfg(unix)]
fn set_permission(file: &File) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = file.metadata()?.permissions();
    permissions.set_mode(0o755);
    file.set_permissions(permissions)?;
}

#[cfg(not(unix))]
fn set_permission(_file: &File) -> Result<()> {
    Ok(())
}