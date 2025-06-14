use serde::Serialize;
use typeshare::typeshare;

use crate::{PlayerId, TeamColor};

#[derive(Debug, Clone, Copy, Serialize)]
#[typeshare]
pub struct TeamPlayerId {
    pub team_color: TeamColor,
    pub player_id: PlayerId,
}

/// Runtime information about the active executor.
#[derive(Debug, Clone, Serialize)]
#[typeshare]
pub struct ExecutorInfo {
    /// Whether the executor is currently paused.
    pub paused: bool,
    /// The player IDs that are currently controlled manually.
    pub manual_controlled_players: Vec<TeamPlayerId>,
    /// Which teams are currently active/controlled.
    pub active_teams: Vec<TeamColor>,
}
