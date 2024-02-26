use anyhow::{anyhow, Context, Result};
use bytes::{Bytes, BytesMut};
use dies_core::workspace_utils::get_workspace_root;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use std::{
    env, fs,
    path::{Path, PathBuf},
};
use tokio::process::Command;

use crate::env_manager::archive::{extract_tar, extract_zip};

use super::venv::Venv;

const PYTHON_RELEASES_URL: &str =
    "https://api.github.com/repos/indygreg/python-build-standalone/releases/tags/";

#[derive(Deserialize, Debug)]
struct Asset {
    name: String,
    browser_download_url: String,
}

#[derive(Deserialize, Debug)]
struct Release {
    assets: Vec<Asset>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Platform {
    Windows,
    Linux,
    Mac,
}

impl Default for Platform {
    fn default() -> Self {
        if cfg!(target_os = "windows") {
            Platform::Windows
        } else if cfg!(target_os = "linux") {
            Platform::Linux
        } else if cfg!(target_os = "macos") {
            Platform::Mac
        } else {
            panic!("Unsupported platform")
        }
    }
}

impl ToString for Platform {
    fn to_string(&self) -> String {
        match self {
            Platform::Windows => "pc-windows-msvc-shared".to_string(),
            Platform::Linux => "unknown-linux-gnu".to_string(),
            Platform::Mac => "apple-darwin".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Arch {
    X86_64,
    Aarch64,
    X86,
}

impl Default for Arch {
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            Arch::X86_64
        } else if cfg!(target_arch = "aarch64") {
            Arch::Aarch64
        } else if cfg!(target_arch = "x86") {
            Arch::X86
        } else {
            panic!("Unsupported architecture")
        }
    }
}

impl ToString for Arch {
    fn to_string(&self) -> String {
        match self {
            Arch::X86_64 => "x86_64".to_string(),
            Arch::Aarch64 => "aarch64".to_string(),
            Arch::X86 => "i386".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PythonDistroConfig {
    pub version: String,
    pub build: u32,
    pub platform: Platform,
    pub arch: Arch,
}

impl PythonDistroConfig {
    pub fn from_version_and_build(version: &str, build: u32) -> Self {
        Self {
            version: version.to_string(),
            build,
            platform: Platform::default(),
            arch: Arch::default(),
        }
    }
}

/// A handle for dealing with an installed Python distribution. Typically, you would use
/// this to obtain a [`Venv`] which can manage packages.
pub struct PythonDistro {
    path: PathBuf,
}

impl PythonDistro {
    /// Create a new instance of [`PythonDistro`], ensuring that we have the correct
    /// build downloaded as specified in the config.
    pub async fn new(config: PythonDistroConfig) -> Result<Self> {
        let path = get_python_dir();
        download_python_distro(config.clone(), path.clone())
            .await
            .context(format!(
                "Failed to download and extract Python distro {}-{}",
                config.version, config.build
            ))?;

        Ok(Self { path })
    }

    /// Get the path to the python binary
    fn python_bin(&self) -> PathBuf {
        if cfg!(target_os = "windows") {
            self.path.join("python.exe")
        } else {
            self.path.join("bin").join("python3")
        }
        // self.path.join("bin").join("python3")
    }

    /// Create a virtual environment in the given directory. The virtual environment
    /// will be placed in a directory named `.venv` in the target directory.
    ///
    /// # Errors
    ///
    /// This function will return an error if the `python3 -m venv` command fails.
    pub async fn create_venv(&self, target_dir: &Path) -> Result<Venv> {
        let venv_dir = target_dir.join(".venv");
        if venv_dir.exists() {
            tracing::debug!("Venv already exists, skipping");
            return Ok(Venv::from_venv_path(venv_dir).await?);
        }

        tracing::info!("Creating new venv...");
        print!("This is the python bin: {:?}", self.python_bin());
        let cmd = Command::new(self.python_bin())
            .current_dir(&target_dir)
            .arg("-m")
            .arg("venv")
            .arg(".venv")
            .output()
            .await
            .context("Failed to create venv")?;

        if !cmd.status.success() {
            tracing::error!("Failed to create venv");
            tracing::error!("stdout: {}", String::from_utf8_lossy(&cmd.stdout));
            tracing::error!("stderr: {}", String::from_utf8_lossy(&cmd.stderr));
            anyhow::bail!("Failed to create venv");
        }

        Ok(Venv::from_venv_path(venv_dir).await?)
    }
}

/// Download and extract the Python distribution
async fn download_python_distro(config: PythonDistroConfig, py_dir: PathBuf) -> Result<()> {
    let distro_file = py_dir.join(".python_distro");
    if distro_file.exists() {
        let existing_config: PythonDistroConfig =
            serde_json::from_reader(&fs::File::open(distro_file.clone())?)?;
        if existing_config == config {
            tracing::debug!("Python distro already installed");
            return Ok(());
        }

        fs::remove_dir_all(&py_dir)?;
        fs::create_dir(&py_dir)?; // Recreate the directory
    }

    tracing::info!(
        "Python ({}-{}) not found, downloading",
        config.version,
        config.build
    );

    let url = PYTHON_RELEASES_URL.to_owned() + config.build.to_string().as_str();
    let client = reqwest::Client::builder()
        .user_agent("Mozilla/5.0 Gecko/20100101 Firefox/122.0")
        .build()?;

    let release: Release = client
        .get(url)
        .send()
        .await
        .context(format!("Failed to get release {}", config.build))?
        .json()
        .await?;

    let build_name = format!(
        "cpython-{}+{}-{}-{}-install_only",
        config.version,
        config.build,
        config.arch.to_string(),
        config.platform.to_string(),
    );
    let asset = release
        .assets
        .iter()
        .find(|a| {
            a.name.contains(&build_name)
                && (a.name.ends_with(".tar.zst")
                    || a.name.ends_with(".tar.gz")
                    || a.name.ends_with(".zip"))
        })
        .context(format!("No asset found for build name {build_name}"))?;
    let archive_name = asset.name.clone();
    let download_url = asset.browser_download_url.clone();

    tracing::debug!("Downloading Python distro from {}", download_url);
    let archive_resp = client.get(download_url).send().await?;
    let total_size = archive_resp.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut stream = archive_resp.bytes_stream();
    let mut archive = BytesMut::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to download python distro")?;
        archive.extend_from_slice(&chunk);

        if total_size > 0 {
            let prev_progress = ((downloaded as f64 / total_size as f64) * 100.0).round();
            downloaded += chunk.len() as u64;
            let progress = ((downloaded as f64 / total_size as f64) * 100.0).round();
            if progress % 10.0 == 0.0 && prev_progress % 10.0 != 0.0 {
                tracing::info!("Progress: {:.0}%", progress);
            }
        }
    }
    let archive = Bytes::from(archive);

    tracing::info!("Finsihed downloading Python distro, extracting...");
    let (extract_tx, extract_rx) = tokio::sync::oneshot::channel();
    let bin_dir_clone = py_dir.clone();
    let task = tokio::task::spawn_blocking(move || {
        let res = if archive_name.ends_with(".tar.zst") {
            extract_tar(archive, &bin_dir_clone, Some("zstd"), "python")
        } else if archive_name.ends_with(".tar.gz") {
            extract_tar(archive, &bin_dir_clone, Some("gz"), "python")
        } else if archive_name.ends_with(".zip") {
            extract_zip(archive, &bin_dir_clone, "python")
        } else {
            Err(anyhow!("Unsupported archive format"))
        };
        extract_tx.send(res).unwrap();
    });
    extract_rx
        .await
        .unwrap()
        .context("Failed to extract python archive")?;
    task.await.unwrap();

    // Write a marker file to indicate the distro
    serde_json::to_writer(&fs::File::create(distro_file)?, &config)?;
    tracing::info!("Python distro installed");

    Ok(())
}

/// Get the directory where the python binaries are stored
///
/// # Panics
/// Panics if the directory does not exist and cannot be created
fn get_python_dir() -> PathBuf {
    // Check the environment variable first
    if let Ok(dir_from_env) = env::var("DIES_PYTHON_BIN_DIR") {
        if let Ok(path) = PathBuf::from(dir_from_env).canonicalize() {
            if !path.exists() {
                fs::create_dir_all(&path).expect("Failed to create .python_bin directory");
            }
            return path;
        }
    }

    // If not set in environment, check the platform's appropriate cache directory
    if let Some(mut cache_dir) = dirs::cache_dir() {
        cache_dir.push("dies");
        cache_dir.push("python");
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir).expect("Failed to create .python_bin directory");
        }
        return cache_dir;
    }

    // If all else fails, use {workspace}/.rye
    let workspace_root = get_workspace_root();
    let path = workspace_root.join(".python_bin");
    if !path.exists() {
        fs::create_dir_all(&path).expect("Failed to create .python_bin directory");
    }
    path
}

#[cfg(test)]
mod test {
    use tempfile::tempdir;

    use super::*;

    #[test_log::test]
    fn test_get_bin_dir() {
        let path = get_python_dir();
        tracing::debug!("Python bin dir: {}", path.display());
        assert!(path.exists());
    }

    #[test_log::test(tokio::test)]
    #[ignore = "This test downloads a large file"]
    async fn test_download_and_extract_python() {
        let config = PythonDistroConfig::from_version_and_build("3.10.13", 20240107);
        download_python_distro(config, get_python_dir())
            .await
            .unwrap();
    }

    #[test_log::test(tokio::test)]
    #[ignore = "This test downloads a large file"]
    async fn test_create_venv() {
        let config = PythonDistroConfig::from_version_and_build("3.10.13", 20240107);
        let distro = PythonDistro::new(config).await.unwrap();
        let target_dir = tempdir().unwrap();
        distro.create_venv(&target_dir.path()).await.unwrap();
        assert!(target_dir.path().join(".venv").exists());
    }
}
