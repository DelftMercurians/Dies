use anyhow::Result;
use dies_core::workspace_utils::get_workspace_root;
use serde::Deserialize;
use tar::Archive;
use zstd::stream::read::Decoder;

use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter},
    path::PathBuf,
    // process::Command,
};

#[derive(Deserialize, Debug)]

struct Release {
    assets: Vec<Asset>,
}

#[derive(Deserialize, Debug)]
struct Asset {
    browser_download_url: String,
}

const DOWNLOAD_URL: &str =
    "https://api.github.com/repos/indygreg/python-build-standalone/releases/tags/20240107";

pub fn get_or_download_python() -> Result<PathBuf> {
    let custom_rye_dir = get_custom_rye_dir()?;
    let path = if cfg!(windows) {
        custom_rye_dir.join("python.exe")
    } else {
        custom_rye_dir.join("python")
    };

    if path.exists() {
        log::info!("Found python binary at {}", path.display());
        return Ok(path);
    }

    log::info!("Downloading python binary to {}", path.display());

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

    // Parse the JSON response
    let release: Release = serde_json::from_str(&release_response.text()?)?;

    // Get the first asset
    let asset = release
        .assets
        .first()
        .ok_or(anyhow::anyhow!("No assets found for the release"))?;

    // Download the asset
    let asset_url = &asset.browser_download_url;
    println!("asset_url: {}", asset_url);

    let asset_response = reqwest::blocking::get(asset_url)?;
    let bytes = asset_response.bytes()?;

    let zstd_path = if cfg!(windows) {
        custom_rye_dir.join("python.tar.zstd")
    } else {
        custom_rye_dir.join("python")
    };

    let file = File::create(zstd_path.clone())?;
    let mut writer = BufWriter::new(&file);
    io::copy(&mut &bytes[..], &mut writer)?;

    println!("zstd_path: {}", zstd_path.display());
    // tar -axvf C:\Users\teodo\AppData\Local\dies\custom_rye\python.tar.zstd
    // tar xvf C:\Users\teodo\AppData\Local\dies\custom_rye\python.tar.zstd
    // tar xvf C:\Users\teodo\Downloads\cpython-3.10.13+20240107-aarch64-apple-darwin-debug-full.tar.zst
    // tar xvf C:\Users\teodo\Downloads\cpython-3.10.13+20240107-aarch64-apple-darwin-debug-full.tar.zst



    // todo!("extract the tar.zstd file");

    let input_file = fs::File::open(zstd_path)?;
    let decoder = Decoder::new(input_file)?;

    // Create a tar archive from the zstd decoder
    let mut archive = Archive::new(decoder);

    let output_folder_path = custom_rye_dir.clone();

    // Extract the contents of the archive to the output folder
    archive.unpack(output_folder_path)?;

    // // Use the tar command to extract the file
    // let output = Command::new("tar")
    //     .arg("-axvf")
    //     .arg(zstd_path.clone())
    //     .output()
    //     .expect("Failed to execute command");

    // // Check if the command was successful
    // if output.status.success() {
    //     println!("Extraction successful!");
    // } else {
    //     // Print the error message if the command failed
    //     eprintln!("Error: {}", String::from_utf8_lossy(&output.stderr));
    // }

    log::info!("Downloaded python binary to {}", path.display());
    Ok(path)
}

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
