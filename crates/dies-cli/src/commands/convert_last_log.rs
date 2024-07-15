use std::{fs, io, path::PathBuf};

use anyhow::Result;

use super::convert_logs::convert_log;

pub fn convert_last_log() -> Result<()> {
    // Read log files from ./logs and find the latest one based on the timestamp
    let (_, input) = fs::read_dir("logs")?
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                entry.metadata().ok().and_then(|metadata| {
                    metadata
                        .modified()
                        .ok()
                        .map(|modified| (modified, entry.path()))
                })
            })
        })
        .max_by_key(|(modified, _)| *modified)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "No log files found"))?;
    let output = PathBuf::from("log.json");
    
    convert_log(&input, &output)
}
