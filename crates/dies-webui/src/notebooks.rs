//! Server-side notebook store: notebooks (cells, title, picked robot) and the
//! latest run state + outputs of every cell, so a page refresh or another
//! laptop sees the same notebooks and results.
//!
//! One `<id>.json` (the notebook) and one `<id>.runs.json` (cell runs) per
//! notebook under a `.dies-notebooks/` dir next to the settings file. Runs are
//! recorded here from the executor's notebook events, not by the browser.

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::Mutex,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use dies_core::{NotebookEvent, NotebookRobot};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

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

#[derive(Debug, Clone, Serialize)]
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
