//! Wire types for the interactive notebook (UI ⇄ executor).

use serde::{Deserialize, Serialize};
use ts_rs::TS;

use crate::{PlayerId, TeamColor};

/// A robot picked in the notebook UI (what `robot()` resolves to).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, TS)]
pub struct NotebookRobot {
    pub team: TeamColor,
    pub id: PlayerId,
}

/// A notebook command from the UI.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(optional_fields)]
#[serde(tag = "type", content = "data")]
pub enum NotebookCmd {
    /// Evaluate a cell. `source` is plain JS, already transformed by the UI to
    /// start its body via `__nb.run(cell, ...)`; `original` is the source as
    /// typed, recorded in the log.
    Eval {
        notebook: String,
        cell: String,
        source: String,
        original: String,
        /// The notebook's picked robot, if any (what `robot()` resolves to).
        robot: Option<NotebookRobot>,
    },
    /// Abort every running and queued cell in every notebook; release all robots.
    Abort,
    /// Abort one notebook and drop its scope (globals, held commands).
    Reset { notebook: String },
}

/// Something a cell produced, or a change in its run state, sent to the UI.
#[derive(Debug, Clone, Serialize, Deserialize, TS)]
#[ts(optional_fields)]
pub struct NotebookEvent {
    /// Notebook id, or `None` for engine-wide events.
    pub notebook: Option<String>,
    /// Cell id, or `None` for notebook-wide events (engine errors, aborts).
    pub cell: Option<String>,
    /// `queued` | `started` | `show` | `done` | `error` | `aborted` | `reset`.
    pub kind: String,
    /// JSON-encoded payload (`show` data, error message).
    pub data: Option<String>,
}
