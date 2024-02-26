use anyhow::{Context, Result};
use bytes::Bytes;
use flate2::read::GzDecoder;
use path_clean::PathClean;
use std::fs::{self, File};
use std::io::{self};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

pub fn extract_zip(archive: Bytes, target_dir: &Path, prefix: &str) -> Result<()> {
    let cursor = Cursor::new(archive.as_ref());
    let mut archive = zip::ZipArchive::new(cursor).context("Failed to create the archive")?;

    let tmp_dir = tempdir().context("Failed to create tempdir")?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).context("Failed to get file from archive by index")?;
        let path = file
            .enclosed_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;
        let path = strip_prefix(&path, prefix).context("Failed to strip the prefix from the path")?;
        let path = tmp_dir.path().join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create all the dirs for the path")?;
        }
        let mut output = File::create(path).context("Failed to create the file")?;
        io::copy(&mut file, &mut output).context("Failed to coppy the file")?;
    }

    // Move the files to the target dir
    for entry in fs::read_dir(tmp_dir.path()).context("Failed to read the files from the temp dir")? {
        let entry = entry.context("Failed to get the entry form the dir")?;
        let path = entry.path();
        let path = target_dir.join(path.file_name().unwrap());
        fs::rename(entry.path(), path).context("Failed to rename (move) the entry to the target dir")?;
    }

    Ok(())
}

pub fn extract_tar(
    archive: Bytes,
    target_dir: &Path,
    compression: Option<&str>,
    prefix: &str,
) -> Result<()> {
    let reader: Box<dyn Read> = match compression {
        Some("gz") => Box::new(GzDecoder::new(archive.as_ref())),
        Some("zstd") => Box::new(zstd::stream::Decoder::new(archive.as_ref()).context("Failed to create decoder for the zstd format")?),
        _ => return Err(anyhow::anyhow!("Unsupported compression format")),
    };

    let tmp_dir = tempdir().context("Failed to get the temp dir")?;
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries().context("Failed to get the entries from the archive")? {
        let mut entry = entry.context("Failed to get the entry")?;
        let path = entry.path().context("Failed to get the path of the entry")?;
        let path = strip_prefix(&path, prefix).context("Failed to Remove the prefix from the path")?;
        let path = tmp_dir.path().join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).context("Failed to create all the dirs for the path")?;
        }
        entry.unpack(&path).context("Failed to unpack the entry")?;
    }

    // Move the files to the target dir
    for entry in fs::read_dir(tmp_dir.path()).context("Failed to read the files from the temp dir")? {
        let entry = entry.context("Failed to get the entry from the dir")?;
        let path = entry.path();
        let path = target_dir.join(path.file_name().unwrap());
        fs::rename(entry.path(), path).context("Failed to rename (move) the entry to the target direcotry")?;
    }

    Ok(())
}

fn strip_prefix<'a>(path: &'a Path, prefix: &str) -> Result<PathBuf> {
    let clean_path = path.clean();
    if let Ok(stripped) = clean_path.strip_prefix(prefix) {
        Ok(stripped.to_path_buf())
    } else {
        Ok(clean_path)
    }
}
