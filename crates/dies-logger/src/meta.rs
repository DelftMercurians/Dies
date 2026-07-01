//! `meta.json` — the once-per-log sidecar holding everything that doesn't
//! belong in a per-frame table: format version, session wall-clock start (to
//! absolutize relative timestamps), the static field geometry, and run config.

use std::io::Read;
use std::path::Path;

use anyhow::Result;
use dies_core::FieldGeometry;
use serde::{Deserialize, Serialize};

/// Bump when the on-disk schema changes incompatibly.
pub const FORMAT_VERSION: u32 = 1;

/// The metadata file name inside a log directory.
pub const META_FILE: &str = "meta.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaJson {
    pub format_version: u32,
    /// Wall-clock unix time (seconds) at which logging started. Per-frame
    /// `t_received` is relative to this.
    pub session_start_unix: f64,
    pub is_simulation: bool,
    pub blue_strategy: Option<String>,
    pub yellow_strategy: Option<String>,
    /// Stable string encoding of the initial side assignment (see
    /// `frame::side_assignment_str`).
    pub side_assignment: String,
    /// Filled on the first frame (geometry isn't known when logging starts).
    pub field_geom: Option<FieldGeometry>,
    /// True only for real-match runs (launched via the `match` subcommand). Used
    /// to filter match logs out of the dev/sim/self-play pile.
    #[serde(default)]
    pub is_match: bool,
    /// GC-reported team names, patched in once known (empty until the GC operator
    /// types them, so usually not available at log-open time).
    #[serde(default)]
    pub blue_team_name: Option<String>,
    #[serde(default)]
    pub yellow_team_name: Option<String>,
    /// Final session stats, patched in on close (None while the log is open).
    #[serde(default)]
    pub frame_count: Option<u64>,
    #[serde(default)]
    pub first_t: Option<f64>,
    #[serde(default)]
    pub last_t: Option<f64>,
}

impl MetaJson {
    pub fn new(
        session_start_unix: f64,
        is_simulation: bool,
        blue_strategy: Option<String>,
        yellow_strategy: Option<String>,
        side_assignment: String,
    ) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            session_start_unix,
            is_simulation,
            blue_strategy,
            yellow_strategy,
            side_assignment,
            field_geom: None,
            is_match: false,
            blue_team_name: None,
            yellow_team_name: None,
            frame_count: None,
            first_t: None,
            last_t: None,
        }
    }

    /// Duration in seconds, if the log was closed cleanly.
    pub fn duration_s(&self) -> Option<f64> {
        match (self.first_t, self.last_t) {
            (Some(a), Some(b)) => Some((b - a).max(0.0)),
            _ => None,
        }
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(dir.join(META_FILE), json)?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(dir.join(META_FILE))?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Read just the `meta.json` member from a `.dieslog` zip (cheap — no full
    /// extraction).
    pub fn read_from_zip(zip_path: &Path) -> Result<Self> {
        let file = std::fs::File::open(zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        let mut entry = archive.by_name(META_FILE)?;
        let mut json = String::new();
        entry.read_to_string(&mut json)?;
        Ok(serde_json::from_str(&json)?)
    }

    /// Read metadata from either a log directory or a `.dieslog` zip.
    pub fn read_any(path: &Path) -> Result<Self> {
        if path.is_dir() {
            Self::read(path)
        } else {
            Self::read_from_zip(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let meta = MetaJson::new(
            1_700_000_000.0,
            true,
            Some("concerto".into()),
            None,
            "yellow_on_positive".into(),
        );
        meta.write(dir.path()).unwrap();
        let read = MetaJson::read(dir.path()).unwrap();
        assert_eq!(read.format_version, FORMAT_VERSION);
        assert_eq!(read.blue_strategy.as_deref(), Some("concerto"));
        assert!(read.is_simulation);
    }
}
