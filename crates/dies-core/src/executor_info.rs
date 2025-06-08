use serde::Serialize;
use typeshare::typeshare;

use crate::{PlayerId, TeamColor};

/// Runtime information about the active executor.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct ExecutorInfo {
    /// Whether the executor is currently paused.
    pub paused: bool,
    /// The player IDs that are currently controlled manually.
    pub manual_controlled_players: Vec<PlayerId>,
    /// Which teams are currently active/controlled.
    pub active_teams: Vec<TeamColor>,
}
