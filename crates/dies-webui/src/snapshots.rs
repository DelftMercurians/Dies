//! Local, gitignored store for simulator field-state snapshots.
//!
//! Each named snapshot is one pretty-printed JSON file under a
//! `.dies-snapshots/` dir next to the settings file. Snapshots capture robot
//! positions + yaw and the ball position (no velocities) so a board state can
//! be saved and replayed by teleporting everything back into place. Unrelated
//! to the scenario/test-driver system.

use std::path::PathBuf;
use std::sync::Arc;

use crate::FieldSnapshot;

#[derive(Clone)]
pub(crate) struct SnapshotStore {
    dir: Arc<PathBuf>,
}

impl SnapshotStore {
    /// Open the store at `dir`, creating it best-effort.
    pub(crate) fn load(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self { dir: Arc::new(dir) }
    }

    /// Names of all stored snapshots, sorted.
    pub(crate) fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(self.dir.as_ref())
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
    }

    pub(crate) fn get(&self, name: &str) -> Option<FieldSnapshot> {
        let path = self.path_for(name)?;
        let contents = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&contents).ok()
    }

    pub(crate) fn save(&self, name: &str, snapshot: &FieldSnapshot) -> bool {
        let Some(path) = self.path_for(name) else {
            log::warn!("Rejected snapshot with unsafe name: {name:?}");
            return false;
        };
        match serde_json::to_string_pretty(snapshot) {
            Ok(s) => {
                if let Err(err) = std::fs::write(&path, s) {
                    log::error!("Failed to write snapshot {name:?}: {err}");
                    return false;
                }
                true
            }
            Err(err) => {
                log::error!("Failed to serialize snapshot {name:?}: {err}");
                false
            }
        }
    }

    pub(crate) fn delete(&self, name: &str) -> bool {
        let Some(path) = self.path_for(name) else {
            return false;
        };
        // Treat "already gone" as success.
        match std::fs::remove_file(&path) {
            Ok(()) => true,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
            Err(err) => {
                log::error!("Failed to delete snapshot {name:?}: {err}");
                false
            }
        }
    }

    /// Resolve a snapshot name to a file path, rejecting anything that could
    /// escape the store directory (path separators, `..`, empty names).
    fn path_for(&self, name: &str) -> Option<PathBuf> {
        let trimmed = name.trim();
        if trimmed.is_empty()
            || trimmed.contains('/')
            || trimmed.contains('\\')
            || trimmed.contains("..")
        {
            return None;
        }
        Some(self.dir.join(format!("{trimmed}.json")))
    }
}
