//! Local, gitignored store for the settings explore/revert UX.
//!
//! Two artifacts live under a `.dies-settings/` dir next to the settings file:
//!   - `baseline.json` — the single user-declared "known good" config.
//!   - `history.jsonl` — an append-only ring of auto-captured snapshots (one
//!     JSON object per line, oldest first), capped to [`MAX_HISTORY`].
//!
//! Edits are *debounced*: the web UI POSTs settings on every keystroke, so we
//! coalesce a burst into a single history entry once the value settles (no
//! further edit for [`DEBOUNCE`]). The store is the only place that decides
//! what's worth remembering; reverting is just re-POSTing a remembered config
//! through the normal settings path, so it needs no special handling here.

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use dies_core::ExecutorSettings;

use crate::{SettingsSnapshot, SettingsSnapshotsResponse};

/// Cap on retained history entries (oldest dropped first).
const MAX_HISTORY: usize = 200;
/// Quiet period after the last edit before it's committed to history.
const DEBOUNCE: Duration = Duration::from_millis(1500);

#[derive(Clone)]
pub(crate) struct SettingsStore {
    shared: Arc<Shared>,
}

struct Shared {
    dir: PathBuf,
    state: Mutex<State>,
}

struct State {
    baseline: Option<SettingsSnapshot>,
    /// Oldest at front, newest at back.
    history: VecDeque<SettingsSnapshot>,
    /// Monotonic counter so a stale debounce timer can tell it's been superseded.
    debounce_gen: u64,
}

impl SettingsStore {
    /// Load the store from `dir`, creating it (and reading any existing
    /// baseline/history) best-effort. Never fails — a missing or corrupt store
    /// just starts empty.
    pub(crate) fn load(dir: PathBuf) -> Self {
        let _ = std::fs::create_dir_all(&dir);

        let baseline = std::fs::read_to_string(dir.join("baseline.json"))
            .ok()
            .and_then(|s| serde_json::from_str::<SettingsSnapshot>(&s).ok());

        let mut history = VecDeque::new();
        if let Ok(contents) = std::fs::read_to_string(dir.join("history.jsonl")) {
            for line in contents.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(snap) = serde_json::from_str::<SettingsSnapshot>(line) {
                    history.push_back(snap);
                }
            }
            while history.len() > MAX_HISTORY {
                history.pop_front();
            }
        }

        Self {
            shared: Arc::new(Shared {
                dir,
                state: Mutex::new(State {
                    baseline,
                    history,
                    debounce_gen: 0,
                }),
            }),
        }
    }

    /// Note a live edit. The actual history commit is deferred until the value
    /// settles, so rapid slider/keystroke bursts produce a single entry.
    pub(crate) fn note_edit(&self, settings: ExecutorSettings) {
        let gen = {
            let mut st = self.shared.state.lock().unwrap();
            st.debounce_gen += 1;
            st.debounce_gen
        };
        let store = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(DEBOUNCE).await;
            store.commit_edit_if_current(gen, settings);
        });
    }

    fn commit_edit_if_current(&self, gen: u64, settings: ExecutorSettings) {
        let history = {
            let mut st = self.shared.state.lock().unwrap();
            // A newer edit arrived during the quiet period — let its timer win.
            if st.debounce_gen != gen {
                return;
            }
            // Don't record a no-op (e.g. a revert back to the latest value).
            if st
                .history
                .back()
                .is_some_and(|last| settings_eq(&last.settings, &settings))
            {
                return;
            }
            st.history.push_back(SettingsSnapshot {
                ts: now_ms(),
                kind: "edit".to_string(),
                settings,
            });
            while st.history.len() > MAX_HISTORY {
                st.history.pop_front();
            }
            chronological(&st.history)
        };
        self.persist_history(&history);
    }

    /// Mark `settings` as the known-good baseline (and drop a marker in history).
    pub(crate) fn set_baseline(&self, settings: ExecutorSettings) -> SettingsSnapshot {
        let snap = SettingsSnapshot {
            ts: now_ms(),
            kind: "baseline".to_string(),
            settings,
        };
        let history = {
            let mut st = self.shared.state.lock().unwrap();
            st.baseline = Some(snap.clone());
            let dup = st
                .history
                .back()
                .is_some_and(|last| settings_eq(&last.settings, &snap.settings));
            if !dup {
                st.history.push_back(snap.clone());
                while st.history.len() > MAX_HISTORY {
                    st.history.pop_front();
                }
            }
            chronological(&st.history)
        };
        self.persist_baseline(&snap);
        self.persist_history(&history);
        snap
    }

    /// Current baseline + history (newest first) for the UI.
    pub(crate) fn snapshots(&self) -> SettingsSnapshotsResponse {
        let st = self.shared.state.lock().unwrap();
        let mut history = chronological(&st.history);
        history.reverse();
        SettingsSnapshotsResponse {
            baseline: st.baseline.clone(),
            history,
        }
    }

    fn persist_baseline(&self, snap: &SettingsSnapshot) {
        let path = self.shared.dir.join("baseline.json");
        if let Ok(s) = serde_json::to_string_pretty(snap) {
            if let Err(err) = std::fs::write(&path, s) {
                log::error!("Failed to write settings baseline: {err}");
            }
        }
    }

    fn persist_history(&self, entries: &[SettingsSnapshot]) {
        let path = self.shared.dir.join("history.jsonl");
        let mut buf = String::new();
        for e in entries {
            if let Ok(line) = serde_json::to_string(e) {
                buf.push_str(&line);
                buf.push('\n');
            }
        }
        if let Err(err) = std::fs::write(&path, buf) {
            log::error!("Failed to write settings history: {err}");
        }
    }
}

fn chronological(history: &VecDeque<SettingsSnapshot>) -> Vec<SettingsSnapshot> {
    history.iter().cloned().collect()
}

fn settings_eq(a: &ExecutorSettings, b: &ExecutorSettings) -> bool {
    serde_json::to_vec(a).ok() == serde_json::to_vec(b).ok()
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
