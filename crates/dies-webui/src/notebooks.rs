//! Server-side notebook store: notebooks (cells, title, picked robot) and the
//! latest run state + outputs of every cell, so a page refresh or another
//! laptop sees the same notebooks and results.
//!
//! One `<id>.json` (the notebook) and one `<id>.runs.json` (cell runs) per
//! notebook under a `.dies-notebooks/` dir next to the settings file. Runs are
//! recorded here from the executor's notebook events, not by the browser.
//!
//! When a log closes, every notebook that ran a cell during it is attached to
//! the log ([`LOG_ATTACHMENT`]), so sharing the log shares the notebook too.
//! Opening such a notebook imports it back here as a separate copy.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dies_core::{NotebookEvent, NotebookRobot};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

/// Name of the notebook snapshot inside a log (a JSON `Vec<NotebookDoc>`).
pub(crate) const LOG_ATTACHMENT: &str = "notebooks.json";

/// Outputs bigger than this are not kept (plots carry their data).
const MAX_STORED_OUTPUT_BYTES: usize = 2 * 1024 * 1024;
/// Minimum interval between run-file writes while cells produce output.
const RUN_WRITE_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct NotebookCell {
    pub id: String,
    pub code: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct Notebook {
    pub id: String,
    pub title: String,
    pub cells: Vec<NotebookCell>,
    /// What `robot()` resolves to; `None` = the only robot on the field.
    #[serde(default)]
    pub robot: Option<NotebookRobot>,
    #[typeshare(serialized_as = "number")]
    #[serde(default)]
    pub updated_ms: i64,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct NotebookSummary {
    pub id: String,
    pub title: String,
    #[typeshare(serialized_as = "number")]
    pub updated_ms: i64,
}

/// One cell output, as carried by the notebook event (`data` is the event's
/// payload: JSON for `show`, an error/abort message otherwise).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct CellOutputState {
    pub kind: String,
    pub data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[typeshare]
pub struct CellRunState {
    /// `queued` | `running` | `done` | `error` | `aborted`.
    pub status: String,
    pub outputs: Vec<CellOutputState>,
    #[typeshare(serialized_as = "number")]
    pub started_ms: Option<i64>,
    #[typeshare(serialized_as = "number")]
    pub finished_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct NotebookDoc {
    pub notebook: Notebook,
    pub runs: HashMap<String, CellRunState>,
}

struct RunsEntry {
    runs: HashMap<String, CellRunState>,
    dirty: bool,
    last_write: Option<Instant>,
}

pub(crate) struct NotebookStore {
    dir: PathBuf,
    runs: Mutex<HashMap<String, RunsEntry>>,
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// FNV-1a, for stable ids of imported notebooks.
fn fnv1a(parts: &[&str]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for b in part.bytes().chain([0]) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

fn safe_id(id: &str) -> Option<&str> {
    let ok = !id.is_empty()
        && id.len() <= 64
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    ok.then_some(id)
}

impl NotebookStore {
    pub(crate) fn load(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);
        Self {
            dir,
            runs: Mutex::new(HashMap::new()),
        }
    }

    fn notebook_path(&self, id: &str) -> Option<PathBuf> {
        safe_id(id).map(|id| self.dir.join(format!("{id}.json")))
    }

    fn runs_path(&self, id: &str) -> Option<PathBuf> {
        safe_id(id).map(|id| self.dir.join(format!("{id}.runs.json")))
    }

    pub(crate) fn list(&self) -> Vec<NotebookSummary> {
        let mut out: Vec<NotebookSummary> = std::fs::read_dir(&self.dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let id = name.strip_suffix(".json")?;
                if id.ends_with(".runs") {
                    return None;
                }
                let nb = self.read_notebook(id)?;
                Some(NotebookSummary {
                    id: nb.id,
                    title: nb.title,
                    updated_ms: nb.updated_ms,
                })
            })
            .collect();
        out.sort_by(|a, b| b.updated_ms.cmp(&a.updated_ms));
        out
    }

    fn read_notebook(&self, id: &str) -> Option<Notebook> {
        let s = std::fs::read_to_string(self.notebook_path(id)?).ok()?;
        serde_json::from_str(&s).ok()
    }

    pub(crate) fn get(&self, id: &str) -> Option<NotebookDoc> {
        let notebook = self.read_notebook(id)?;
        let mut runs = self.runs.lock().unwrap();
        let entry = self.runs_entry(&mut runs, id);
        Some(NotebookDoc {
            notebook,
            runs: entry.runs.clone(),
        })
    }

    pub(crate) fn save(&self, mut notebook: Notebook) -> Option<Notebook> {
        let path = self.notebook_path(&notebook.id)?;
        notebook.updated_ms = now_ms();
        let s = serde_json::to_string_pretty(&notebook).ok()?;
        if let Err(err) = std::fs::write(&path, s) {
            log::error!("Failed to write notebook {}: {err}", notebook.id);
            return None;
        }
        Some(notebook)
    }

    pub(crate) fn delete(&self, id: &str) -> bool {
        let (Some(nb), Some(runs)) = (self.notebook_path(id), self.runs_path(id)) else {
            return false;
        };
        self.runs.lock().unwrap().remove(id);
        let _ = std::fs::remove_file(runs);
        match std::fs::remove_file(nb) {
            Ok(()) => true,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => true,
            Err(e) => {
                log::error!("Failed to delete notebook {id}: {e}");
                false
            }
        }
    }

    fn runs_entry<'a>(
        &self,
        runs: &'a mut HashMap<String, RunsEntry>,
        id: &str,
    ) -> &'a mut RunsEntry {
        runs.entry(id.to_string()).or_insert_with(|| {
            let loaded = self
                .runs_path(id)
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default();
            RunsEntry {
                runs: loaded,
                dirty: false,
                last_write: None,
            }
        })
    }

    /// Record a notebook event (from the executor) into the cell's run state.
    pub(crate) fn on_event(&self, ev: &NotebookEvent) {
        let Some(nb) = ev.notebook.as_deref().and_then(safe_id) else {
            return;
        };
        let mut all = self.runs.lock().unwrap();
        let entry = self.runs_entry(&mut all, nb);
        let now = now_ms();
        let mut terminal = false;
        match (&ev.cell, ev.kind.as_str()) {
            (None, "reset") => {
                for run in entry.runs.values_mut() {
                    if run.status == "running" || run.status == "queued" {
                        run.status = "aborted".into();
                        run.finished_ms = Some(now);
                    }
                }
                terminal = true;
            }
            (Some(cell), kind) => {
                let run = entry.runs.entry(cell.clone()).or_default();
                match kind {
                    "queued" => {
                        *run = CellRunState {
                            status: "queued".into(),
                            outputs: vec![],
                            started_ms: Some(now),
                            finished_ms: None,
                        }
                    }
                    "started" => {
                        run.status = "running".into();
                        run.started_ms = Some(now);
                    }
                    "show" => {
                        let data = match &ev.data {
                            Some(d) if d.len() > MAX_STORED_OUTPUT_BYTES => Some(
                                serde_json::json!({
                                    "type": "text",
                                    "text": format!(
                                        "(output too large to keep: {:.1} MB — re-run to see it)",
                                        d.len() as f64 / 1e6
                                    ),
                                })
                                .to_string(),
                            ),
                            d => d.clone(),
                        };
                        run.outputs.push(CellOutputState {
                            kind: "show".into(),
                            data,
                        });
                    }
                    "done" | "error" | "aborted" => {
                        run.status = kind.into();
                        if kind != "done" {
                            run.outputs.push(CellOutputState {
                                kind: kind.into(),
                                data: ev.data.clone(),
                            });
                        }
                        run.finished_ms = Some(now);
                        terminal = true;
                    }
                    _ => {}
                }
            }
            _ => return,
        }
        entry.dirty = true;
        let due = entry
            .last_write
            .map_or(true, |t| t.elapsed() >= RUN_WRITE_INTERVAL);
        if terminal || due {
            self.write_runs(nb, entry);
        }
    }

    /// Every notebook with a cell run started at or after `since_ms`, with only
    /// those runs. This is what gets attached to a log when it closes.
    pub(crate) fn snapshot_since(&self, since_ms: i64) -> Vec<NotebookDoc> {
        let all = self.runs.lock().unwrap();
        let mut docs: Vec<NotebookDoc> = all
            .iter()
            .filter_map(|(id, entry)| {
                let runs: HashMap<String, CellRunState> = entry
                    .runs
                    .iter()
                    .filter(|(_, r)| r.started_ms.is_some_and(|t| t >= since_ms))
                    .map(|(cell, r)| {
                        let mut r = r.clone();
                        // Still going when the log closed: it was cut off.
                        if r.status == "running" || r.status == "queued" {
                            r.status = "aborted".into();
                        }
                        (cell.clone(), r)
                    })
                    .collect();
                if runs.is_empty() {
                    return None;
                }
                let notebook = self.read_notebook(id)?;
                Some(NotebookDoc { notebook, runs })
            })
            .collect();
        docs.sort_by(|a, b| a.notebook.title.cmp(&b.notebook.title));
        docs
    }

    /// Store a notebook from a log as a new notebook (fresh notebook and cell
    /// ids, so it never mixes with the original). Opening the same one again
    /// returns the existing copy instead of making another.
    pub(crate) fn import(&self, log_name: &str, doc: NotebookDoc) -> Option<NotebookSummary> {
        let id = format!("log-{:016x}", fnv1a(&[log_name, &doc.notebook.id]));
        if let Some(nb) = self.read_notebook(&id) {
            return Some(NotebookSummary {
                id: nb.id,
                title: nb.title,
                updated_ms: nb.updated_ms,
            });
        }
        let mut runs = HashMap::new();
        let cells = doc
            .notebook
            .cells
            .into_iter()
            .enumerate()
            .map(|(i, cell)| {
                let new_id = format!("{id}-c{i}");
                if let Some(run) = doc.runs.get(&cell.id) {
                    runs.insert(new_id.clone(), run.clone());
                }
                NotebookCell {
                    id: new_id,
                    code: cell.code,
                }
            })
            .collect();
        let saved = self.save(Notebook {
            id: id.clone(),
            title: format!("{} ({log_name})", doc.notebook.title),
            cells,
            robot: doc.notebook.robot,
            updated_ms: 0,
        })?;
        let mut all = self.runs.lock().unwrap();
        let entry = self.runs_entry(&mut all, &id);
        entry.runs = runs;
        self.write_runs(&id, entry);
        Some(NotebookSummary {
            id: saved.id,
            title: saved.title,
            updated_ms: saved.updated_ms,
        })
    }

    /// Write any run state still pending (called periodically).
    pub(crate) fn flush(&self) {
        let mut all = self.runs.lock().unwrap();
        let ids: Vec<String> = all
            .iter()
            .filter(|(_, e)| e.dirty)
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            if let Some(entry) = all.get_mut(&id) {
                self.write_runs(&id, entry);
            }
        }
    }

    fn write_runs(&self, id: &str, entry: &mut RunsEntry) {
        let Some(path) = self.runs_path(id) else {
            return;
        };
        match serde_json::to_string(&entry.runs) {
            Ok(s) => {
                if let Err(e) = std::fs::write(path, s) {
                    log::error!("Failed to write notebook runs {id}: {e}");
                }
            }
            Err(e) => log::error!("Failed to serialize notebook runs {id}: {e}"),
        }
        entry.dirty = false;
        entry.last_write = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(cell: &str, kind: &str, data: Option<&str>) -> NotebookEvent {
        NotebookEvent {
            notebook: Some("n1".into()),
            cell: Some(cell.into()),
            kind: kind.into(),
            data: data.map(String::from),
        }
    }

    #[test]
    fn snapshot_and_import_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dies-nb-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = NotebookStore::load(dir.clone());
        store
            .save(Notebook {
                id: "n1".into(),
                title: "Swerve".into(),
                cells: vec![
                    NotebookCell {
                        id: "a".into(),
                        code: "1".into(),
                    },
                    NotebookCell {
                        id: "b".into(),
                        code: "2".into(),
                    },
                ],
                robot: None,
                updated_ms: 0,
            })
            .unwrap();
        assert!(store.snapshot_since(0).is_empty());

        store.on_event(&ev("a", "queued", None));
        store.on_event(&ev("a", "started", None));
        store.on_event(&ev("a", "show", Some(r#"{"type":"text","text":"hi"}"#)));
        store.on_event(&ev("a", "done", None));
        store.on_event(&ev("b", "queued", None));
        store.on_event(&ev("b", "started", None));

        let snap = store.snapshot_since(0);
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].runs.len(), 2);
        assert_eq!(
            snap[0].runs["b"].status, "aborted",
            "cut-off run marked aborted"
        );
        assert!(
            store.snapshot_since(i64::MAX).is_empty(),
            "older runs are left out"
        );

        let json = serde_json::to_vec(&snap).unwrap();
        let doc: Vec<NotebookDoc> = serde_json::from_slice(&json).unwrap();
        let first = store.import("dies-x", doc[0].clone()).unwrap();
        assert_eq!(first.title, "Swerve (dies-x)");
        let again = store.import("dies-x", doc[0].clone()).unwrap();
        assert_eq!(first.id, again.id, "opening twice reuses the copy");

        let copy = store.get(&first.id).unwrap();
        assert!(copy
            .notebook
            .cells
            .iter()
            .all(|c| c.id != "a" && c.id != "b"));
        let a = &copy.runs[&copy.notebook.cells[0].id];
        assert_eq!(a.status, "done");
        assert_eq!(a.outputs.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
