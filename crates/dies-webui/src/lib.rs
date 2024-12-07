use std::{collections::HashMap, path::PathBuf};

use dies_basestation_client::BasestationHandle;
use dies_core::{
    DebugMap, ExecutorInfo, ExecutorSettings, PlayerFeedbackMsg, PlayerId, PlayerOverrideCommand,
    ScenarioInfo, SimulatorCmd, WorldFrame,
};
use dies_ssl_client::SslClientConfig;
use dies_team::scenarios::ScenarioType;
use serde::{Deserialize, Serialize};
use typeshare::typeshare;

mod executor_task;
mod interactive_main_loop;
mod routes;
mod server;
mod ui_common;

pub use server::start;

/// The configuration for the web UI.
#[derive(Debug, Clone)]
pub struct UiConfig {
    pub port: u16,
    pub settings_file: PathBuf,
    pub environment: UiEnvironment,
    pub start_mode: UiMode,
    pub start_scenario: Option<String>,
}

#[derive(Debug, Clone)]
pub enum UiEnvironment {
    WithLive {
        ssl_config: SslClientConfig,
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
    StartingScenario(ScenarioInfo),
    RunningExecutor { scenario: String },
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
        player_id: PlayerId,
        manual_override: bool,
    },
    OverrideCommand {
        player_id: PlayerId,
        command: PlayerOverrideCommand,
    },
    SimulatorCmd(SimulatorCmd),
    SetPause(bool),
    StartScenario {
        scenario: ScenarioType,
    },
    GcCommand(String),
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
    Loaded(WorldFrame),
    None,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[typeshare]
pub(crate) enum WsMessage<'a> {
    WorldUpdate(&'a WorldFrame),
    Debug(&'a DebugMap),
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

#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub(crate) struct BasestationResponse {
    pub(crate) players: HashMap<PlayerId, PlayerFeedbackMsg>,
}
