use anyhow::Result;
use bytes::Bytes;
use flate2::read::GzDecoder;
use path_clean::PathClean;
use std::env::temp_dir;
use std::fs::{self, File};
use std::io::{self, BufReader};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use tempfile::tempdir;

pub fn extract_zip(archive: Bytes, target_dir: &Path, prefix: &str) -> Result<()> {
    let cursor = Cursor::new(archive.as_ref());
    let mut archive = zip::ZipArchive::new(cursor)?;

    let tmp_dir = tempdir()?;
    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let path = file
            .enclosed_name()
            .ok_or_else(|| anyhow::anyhow!("Invalid file name"))?;
        let path = strip_prefix(&path, prefix)?;
        let path = tmp_dir.path().join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = File::create(path)?;
        io::copy(&mut file, &mut output)?;
    }

    // Move the files to the target dir
    for entry in fs::read_dir(tmp_dir.path())? {
        let entry = entry?;
        let path = entry.path();
        let path = target_dir.join(path.file_name().unwrap());
        fs::rename(entry.path(), path)?;
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
        Some("zstd") => Box::new(zstd::stream::Decoder::new(archive.as_ref())?),
        _ => return Err(anyhow::anyhow!("Unsupported compression format")),
    };

    let tmp_dir = tempdir()?;
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let path = strip_prefix(&path, prefix)?;
        let path = tmp_dir.path().join(path);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        entry.unpack(&path)?;
    }

    // Move the files to the target dir
    for entry in fs::read_dir(tmp_dir.path())? {
        let entry = entry?;
        let path = entry.path();
        let path = target_dir.join(path.file_name().unwrap());
        fs::rename(entry.path(), path)?;
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
