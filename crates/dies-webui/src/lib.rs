use std::path::PathBuf;

use dies_basestation_client::BasestationHandle;
use dies_core::{
    Angle, BaseStationInfo, BenchCommand, DebugMap, ExecutorInfo, ExecutorSettings, FieldSnapshot,
    GcSimCommand, ParamValue, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand, SideAssignment,
    SimulatorCmd, TeamColor, TeamConfiguration, Vector2, WorldData, WorldUpdate,
};
use dies_ssl_client::SslClientConfig;
use dies_test_driver::{TestLogEntry, TestStatus};
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

mod bench;
mod executor_task;
mod replay_controller;
mod routes;
mod server;
mod settings_store;
mod snapshots;

pub use server::start;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlledTeam {
    Blue,
    Yellow,
    Both,
}

/// The configuration for the web UI.
#[derive(Debug, Clone)]
pub struct UiConfig {
    pub port: u16,
    pub settings_file: PathBuf,
    pub environment: UiEnvironment,
    pub start_mode: UiMode,
    pub auto_start: bool,
    pub controlled_teams: ControlledTeam,
    pub calibration_mode: bool,
    /// IPC strategy binary name (None = no strategy). Applied to every active team.
    pub strategy: Option<String>,
    /// Per-team strategy override for blue. When set, activates the blue team and
    /// uses this binary instead of [`strategy`](Self::strategy).
    pub blue_strategy: Option<String>,
    /// Per-team strategy override for yellow (see [`blue_strategy`](Self::blue_strategy)).
    pub yellow_strategy: Option<String>,
    /// Dev-only: hot-reload the strategy process when its binary is rebuilt.
    pub hot_reload: bool,
    /// Dev-only: artificial delay (ms) applied to incoming vision packets.
    pub vision_delay_ms: u32,
    /// Directory where session logs are written/browsed for replay.
    pub log_directory: PathBuf,
}

#[derive(Debug, Clone)]
pub enum UiEnvironment {
    WithLive {
        /// Vision/GC config. `None` means no vision was configured (e.g.
        /// `--connection-mode none`): the basestation is still live (so the test
        /// bench works), but a Live executor can only start with `allow_no_vision`.
        ssl_config: Option<SslClientConfig>,
        bs_handle: BasestationHandle,
    },
    SimulationOnly,
}

impl UiConfig {
    pub fn is_live_available(&self) -> bool {
        match self.environment {
            UiEnvironment::WithLive { .. } => true,
            UiEnvironment::SimulationOnly => false,
        }
    }
}

/// The current status of the executor.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub(crate) enum ExecutorStatus {
    None,
    RunningExecutor,
    Failed(String),
}

/// Runtime information about the active executor.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct ExecutorInfoResponse {
    info: Option<ExecutorInfo>,
}

/// The current status of the UI.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct UiStatus {
    pub(crate) is_live_available: bool,
    pub(crate) ui_mode: UiMode,
    pub(crate) executor: ExecutorStatus,
}

/// A command from the frontend to the backend.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub(crate) enum UiCommand {
    SetManualOverride {
        team_color: TeamColor,
        player_id: PlayerId,
        manual_override: bool,
    },
    OverrideCommand {
        team_color: TeamColor,
        player_id: PlayerId,
        command: PlayerOverrideCommand,
    },
    SimulatorCmd(SimulatorCmd),
    SetPause(bool),
    Start,
    GcCommand(GcSimCommand),
    /// Control which teams are active
    SetActiveTeams {
        blue_active: bool,
        yellow_active: bool,
    },
    /// Set side assignment
    SetSideAssignment {
        side_assignment: SideAssignment,
    },
    /// Set complete team configuration (for pre-executor setup)
    SetTeamConfiguration {
        configuration: TeamConfiguration,
    },
    /// Swap team colors (Blue <-> Yellow)
    SwapTeamColors,
    /// Swap team sides (BlueOnPositive <-> YellowOnPositive)
    SwapTeamSides,
    /// Load and start a JS scenario by file name (resolved against `scenarios/`).
    StartScenario {
        scenario: String,
        team: Option<TeamColor>,
    },
    /// Stop the currently running scenario and return to strategy mode.
    StopScenario,
    /// Drop a point-of-interest marker at the current live frame (double-space).
    AddMarker {
        label: Option<String>,
    },
    /// Set a single runtime strategy parameter for a team.
    SetStrategyParam {
        team_color: TeamColor,
        key: String,
        value: ParamValue,
    },
    /// Load a recorded log (directory or `.dieslog` zip) for replay.
    LoadLog {
        path: String,
    },
    /// Replay transport controls.
    ReplayPlay,
    ReplayPause,
    ReplaySeek {
        t: f64,
    },
    ReplaySetSpeed {
        speed: f64,
    },
    /// Direct robot test-bench command (bypasses the executor entirely).
    Bench(BenchCommand),
    Stop,
}

#[derive(Debug, Clone, Deserialize)]
#[typeshare]
pub(crate) struct PostUiCommandBody {
    pub(crate) command: UiCommand,
}

/// The current mode of the UI - either simulation or live.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[typeshare]
pub enum UiMode {
    Simulation,
    Live,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[typeshare]
pub(crate) struct PostUiModeBody {
    pub mode: UiMode,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub(crate) enum UiWorldState {
    Loaded(WorldData),
    None,
}

/// A user point-of-interest marker, surfaced to the replay scrubber.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct ReplayMarker {
    #[typeshare(serialized_as = "number")]
    pub frame_id: u64,
    pub t: f64,
    pub label: Option<String>,
}

/// State of the replay player, pushed to the frontend on change.
#[derive(Debug, Clone, Serialize, Default)]
#[typeshare]
pub(crate) struct ReplayState {
    pub loaded: bool,
    pub playing: bool,
    pub speed: f64,
    pub t_min: f64,
    pub t_max: f64,
    pub current_t: f64,
    #[typeshare(serialized_as = "number")]
    pub current_frame_id: u64,
    #[typeshare(serialized_as = "number")]
    pub frame_count: u64,
    pub markers: Vec<ReplayMarker>,
}

/// A single backend log line streamed to the console panel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct ConsoleLogMessage {
    pub level: ConsoleLogLevel,
    /// Log target (module path).
    pub target: String,
    pub message: String,
    /// Milliseconds since UNIX epoch at emission time.
    #[typeshare(serialized_as = "number")]
    pub ts_ms: i64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[typeshare]
pub enum ConsoleLogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<log::Level> for ConsoleLogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Trace => ConsoleLogLevel::Trace,
            log::Level::Debug => ConsoleLogLevel::Debug,
            log::Level::Info => ConsoleLogLevel::Info,
            log::Level::Warn => ConsoleLogLevel::Warn,
            log::Level::Error => ConsoleLogLevel::Error,
        }
    }
}

/// WebSocket message types sent from backend to frontend
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub(crate) enum WsMessage<'a> {
    /// Carries the full `WorldUpdate` (world data + frame id) so the frontend can
    /// show the current frame number for both live and replay.
    WorldUpdate(&'a WorldUpdate),
    Debug(&'a DebugMap),
    ScenarioLog(&'a TestLogEntry),
    ScenarioStatus(&'a TestStatus),
    ReplayState(&'a ReplayState),
    ConsoleLog(&'a ConsoleLogMessage),
}

/// A recorded log available for replay (directory or `.dieslog` zip).
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct LogInfo {
    pub(crate) name: String,
    pub(crate) path: String,
    pub(crate) session_start_unix: f64,
    pub(crate) duration_s: Option<f64>,
    pub(crate) end_unix: Option<f64>,
    #[typeshare(serialized_as = "number")]
    pub(crate) frame_count: Option<u64>,
    pub(crate) is_simulation: bool,
    pub(crate) blue_strategy: Option<String>,
    pub(crate) yellow_strategy: Option<String>,
    pub(crate) is_zip: bool,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct LogsResponse {
    pub(crate) logs: Vec<LogInfo>,
}

/// A single scenario file available on disk.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct ScenarioInfo {
    pub(crate) name: String,
    pub(crate) path: String,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct ScenariosResponse {
    pub(crate) scenarios: Vec<ScenarioInfo>,
    pub(crate) status: TestStatus,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct GetDebugMapResponse {
    pub(crate) debug_map: DebugMap,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[typeshare]
pub(crate) struct ExecutorSettingsResponse {
    pub(crate) settings: ExecutorSettings,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[typeshare]
pub(crate) struct PostExecutorSettingsBody {
    pub(crate) settings: ExecutorSettings,
}

/// A captured settings configuration with provenance. `kind` is a free-form
/// tag: `"edit"` (debounced auto-snapshot of a live edit), `"baseline"` (the
/// user-declared known-good config), or `"revert"`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[typeshare]
pub(crate) struct SettingsSnapshot {
    /// Unix epoch milliseconds.
    #[typeshare(serialized_as = "number")]
    pub(crate) ts: i64,
    pub(crate) kind: String,
    pub(crate) settings: ExecutorSettings,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct SettingsSnapshotsResponse {
    /// The current known-good config, or null if none has been marked yet.
    pub(crate) baseline: Option<SettingsSnapshot>,
    /// Auto-captured history, newest first.
    pub(crate) history: Vec<SettingsSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct SnapshotsResponse {
    /// Stored snapshot names, sorted.
    pub(crate) names: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[typeshare]
pub(crate) struct SaveSnapshotBody {
    pub(crate) name: String,
    pub(crate) snapshot: FieldSnapshot,
}

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct BasestationResponse {
    pub(crate) blue_team: Vec<PlayerFeedbackMsg>,
    pub(crate) yellow_team: Vec<PlayerFeedbackMsg>,
    pub(crate) unknown_team: Vec<PlayerFeedbackMsg>,
    /// RF basestation status (radios, version, channel), for the test bench.
    pub(crate) base_info: Option<BaseStationInfo>,
}
