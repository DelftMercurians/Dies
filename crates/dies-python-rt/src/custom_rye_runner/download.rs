use anyhow::Result;
use const_format::formatcp;
use dies_core::workspace_utils::get_workspace_root;
use flate2::read::GzDecoder;
use serde::Deserialize;
use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter},
    path::PathBuf,
};

#[derive(Deserialize, Debug)]
struct Release {
    assets: Vec<Asset>,
}

#[derive(Deserialize, Debug)]
struct Asset {
    browser_download_url: String,
}

// const VERSION: &str = "0.15.2";
// const VERSION: &str = "3.12.1"; // latest python version?

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

const TAG: &str = "20240107";
// const DOWNLOAD_URL: &str = formatcp!(
//     "https://api.github.com/repos/indygreg/python-build-standalone/releases/tags/{}",
//     TAG
// );
const DOWNLOAD_URL: &str =
    "https://api.github.com/repos/indygreg/python-build-standalone/releases/latest";

pub fn remove_python() -> Result<()> {
    let rye_dir = get_custom_rye_dir()?;
    let path = if cfg!(windows) {
        rye_dir.join("python.exe")
    } else {
        rye_dir.join("python")
    };

    if path.exists() {
        log::info!("Removing python binary at {}", path.display());
        fs::remove_file(path)?;
    }

    Ok(())
}

pub fn get_or_download_python() -> Result<PathBuf> {
    // GitHub releases API
    let rye_dir = get_custom_rye_dir()?;
    let path = if cfg!(windows) {
        rye_dir.join("python.exe")
    } else {
        rye_dir.join("python")
    };

    if path.exists() {
        log::info!("Found python binary at {}", path.display());
        return Ok(path);
    }

    log::info!("Downloading python binary to {}", path.display());
    println!(
        "Downloading python binary to {}, from: {}",
        path.display(),
        DOWNLOAD_URL
    );

    let release_response = reqwest::blocking::Client::new()
        .get(DOWNLOAD_URL)
        .header("User-Agent", "Delft-Mercurians/dies")
        .send()?;

    // Check if the request was successful (status code 200)
    if !release_response.status().is_success() {
        panic!(
            "GitHub API request failed with status code: {}",
            release_response.status()
        );
    }

    println!("release_response: {:?}", release_response);

    // Parse the JSON response
    let release: Release = serde_json::from_str(&release_response.text()?)?;

    //? Get the first asset - we may need more
    let asset = release
        .assets
        .first()
        .ok_or(anyhow::anyhow!("No assets found for the release"))?;

    println!("asset: {:?}", asset);

    // Download the asset
    let asset_url = &asset.browser_download_url;
    let asset_response = reqwest::blocking::get(asset_url)?;
    let bytes = asset_response.bytes()?;

    let file = File::create(path.clone())?;
    let mut writer = BufWriter::new(&file);
    // if BINARY.ends_with(".gz") {
    //     let mut decoder = GzDecoder::new(&bytes[..]);
    //     io::copy(&mut decoder, &mut writer)?;
    // } else {
    io::copy(&mut &bytes[..], &mut writer)?;
    // }

    // set_permission(&file)?;

    log::info!("Downloaded python binary to {}", path.display());
    Ok(path)
}

pub fn get_custom_rye_dir() -> Result<PathBuf> {
    // Check the environment variable first
    if let Ok(dir_from_env) = env::var("DIES_CUSTOM_RYE_DIR") {
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
        cache_dir.push("custom_rye");
        if !cache_dir.exists() {
            fs::create_dir_all(&cache_dir)?;
        }
        return Ok(cache_dir);
    }

    // If all else fails, use {workspace}/.rye
    let workspace_root = get_workspace_root();
    let path = workspace_root.join(".custom_rye");
    if !path.exists() {
        fs::create_dir_all(&path)?;
    }
    Ok(path)
}
