//! Strategy Host - manages multiple strategy processes.
//!
//! The StrategyHost is the main entry point for the strategy system. It manages
//! strategy processes for both teams, handles IPC, and coordinates with the executor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use dies_core::{
    debug_record, DebugShape, DebugValue, PlayerId, SideAssignment,
    TeamColor, TeamData,
};
use dies_strategy_protocol::{DebugEntry, SkillCommand, SkillStatus, StrategyConfig};
use tracing::{error, info, warn};

use super::connection::{ConnectionError, ConnectionState, StrategyConnection};

/// Configuration for the strategy host.
#[derive(Debug, Clone)]
pub struct StrategyHostConfig {
    /// Directory containing strategy binaries.
    pub strategies_dir: PathBuf,
    /// Strategy binary name for blue team (None = no strategy).
    pub blue_strategy: Option<String>,
    /// Strategy binary name for yellow team (None = no strategy).
    pub yellow_strategy: Option<String>,
    /// Current side assignment.
    pub side_assignment: SideAssignment,
}

impl Default for StrategyHostConfig {
    fn default() -> Self {
        Self {
            strategies_dir: PathBuf::from("target/strategies"),
            blue_strategy: None,
            yellow_strategy: None,
            side_assignment: SideAssignment::BlueOnPositive,
        }
    }
}

/// Output from processing strategies for a frame.
#[derive(Debug, Default)]
pub struct FrameOutput {
    /// Skill commands per player for blue team.
    pub blue_commands: HashMap<PlayerId, Option<SkillCommand>>,
    /// Skill commands per player for yellow team.
    pub yellow_commands: HashMap<PlayerId, Option<SkillCommand>>,
    /// Player roles for blue team.
    pub blue_roles: HashMap<PlayerId, String>,
    /// Player roles for yellow team.
    pub yellow_roles: HashMap<PlayerId, String>,
}

/// Manages strategy processes and their connections.
///
/// The host handles:
/// - Strategy discovery and listing
/// - Spawning strategy processes on demand
/// - Managing connections to each team's strategy
/// - Sending world updates and collecting skill commands
/// - Debug data forwarding
pub struct StrategyHost {
    /// Configuration.
    config: StrategyHostConfig,
    /// Connection for blue team strategy.
    blue_connection: Option<StrategyConnection>,
    /// Connection for yellow team strategy.
    yellow_connection: Option<StrategyConnection>,
    /// Skill statuses for blue team.
    blue_skill_statuses: HashMap<PlayerId, SkillStatus>,
    /// Skill statuses for yellow team.
    yellow_skill_statuses: HashMap<PlayerId, SkillStatus>,
}

impl StrategyHost {
    /// Create a new strategy host.
    pub fn new(config: StrategyHostConfig) -> Self {
        Self {
            config,
            blue_connection: None,
            yellow_connection: None,
            blue_skill_statuses: HashMap::new(),
            yellow_skill_statuses: HashMap::new(),
        }
    }

    /// Get the strategies directory.
    pub fn strategies_dir(&self) -> &Path {
        &self.config.strategies_dir
    }

    /// Discover available strategy binaries.
    pub fn discover_strategies(&self) -> Vec<String> {
        let mut strategies = Vec::new();

        if self.config.strategies_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&self.config.strategies_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        // Check if it's executable
                        #[cfg(unix)]
                        {
                            use std::os::unix::fs::PermissionsExt;
                            if let Ok(meta) = path.metadata() {
                                if meta.permissions().mode() & 0o111 != 0 {
                                    if let Some(name) = path.file_name() {
                                        strategies.push(name.to_string_lossy().to_string());
                                    }
                                }
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            if let Some(name) = path.file_name() {
                                strategies.push(name.to_string_lossy().to_string());
                            }
                        }
                    }
                }
            }
        }

        strategies.sort();
        strategies
    }

    /// Set the strategy for a team.
    ///
    /// If a strategy is already running for the team, it will be stopped first.
    pub fn set_strategy(&mut self, team: TeamColor, strategy_name: Option<String>) {
        // Stop existing connection
        self.stop_strategy(team);

        // Update config
        match team {
            TeamColor::Blue => self.config.blue_strategy = strategy_name.clone(),
            TeamColor::Yellow => self.config.yellow_strategy = strategy_name.clone(),
        }

        // Start new strategy if specified
        if let Some(name) = strategy_name {
            let strategy_path = self.config.strategies_dir.join(&name);
            if strategy_path.exists() {
                self.start_strategy(team, strategy_path);
            } else {
                error!(
                    "Strategy binary not found: {:?} for team {:?}",
                    strategy_path, team
                );
            }
        }
    }

    /// Start a strategy for a team.
    fn start_strategy(&mut self, team: TeamColor, strategy_path: PathBuf) {
        let config = StrategyConfig::default();
        let mut connection = StrategyConnection::new(
            team,
            strategy_path.clone(),
            self.config.side_assignment,
            config,
        );

        match connection.start() {
            Ok(()) => {
                info!(
                    "Started strategy {:?} for team {:?}",
                    strategy_path, team
                );
                match team {
                    TeamColor::Blue => self.blue_connection = Some(connection),
                    TeamColor::Yellow => self.yellow_connection = Some(connection),
                }
            }
            Err(e) => {
                error!(
                    "Failed to start strategy {:?} for team {:?}: {}",
                    strategy_path, team, e
                );
            }
        }
    }

    /// Stop the strategy for a team.
    pub fn stop_strategy(&mut self, team: TeamColor) {
        let connection = match team {
            TeamColor::Blue => self.blue_connection.take(),
            TeamColor::Yellow => self.yellow_connection.take(),
        };

        if let Some(mut conn) = connection {
            if let Err(e) = conn.shutdown() {
                warn!("Error shutting down strategy for team {:?}: {}", team, e);
            }
        }
    }

    /// Update the side assignment.
    pub fn set_side_assignment(&mut self, side_assignment: SideAssignment) {
        self.config.side_assignment = side_assignment;

        if let Some(ref mut conn) = self.blue_connection {
            conn.set_side_assignment(side_assignment);
        }
        if let Some(ref mut conn) = self.yellow_connection {
            conn.set_side_assignment(side_assignment);
        }
    }

    /// Update skill statuses from the executor.
    pub fn update_skill_statuses(
        &mut self,
        team: TeamColor,
        statuses: HashMap<PlayerId, SkillStatus>,
    ) {
        match team {
            TeamColor::Blue => self.blue_skill_statuses = statuses,
            TeamColor::Yellow => self.yellow_skill_statuses = statuses,
        }
    }

    /// Check and accept pending connections.
    ///
    /// Should be called periodically to accept connections from starting strategies.
    pub fn poll_connections(&mut self) {
        // Check blue connection
        if let Some(ref mut conn) = self.blue_connection {
            if conn.state() == ConnectionState::Starting {
                match conn.try_accept() {
                    Ok(true) => {
                        info!("Blue team strategy connected and ready");
                    }
                    Ok(false) => {}
                    Err(e) => {
                        error!("Blue team strategy connection failed: {}", e);
                        self.blue_connection = None;
                    }
                }
            }
        }

        // Check yellow connection
        if let Some(ref mut conn) = self.yellow_connection {
            if conn.state() == ConnectionState::Starting {
                match conn.try_accept() {
                    Ok(true) => {
                        info!("Yellow team strategy connected and ready");
                    }
                    Ok(false) => {}
                    Err(e) => {
                        error!("Yellow team strategy connection failed: {}", e);
                        self.yellow_connection = None;
                    }
                }
            }
        }
    }

    /// Process a frame for both teams.
    ///
    /// Sends world updates to connected strategies and collects skill commands.
    pub fn update(
        &mut self,
        blue_team_data: Option<&TeamData>,
        yellow_team_data: Option<&TeamData>,
    ) -> FrameOutput {
        // Check for pending connections
        self.poll_connections();

        let mut output = FrameOutput::default();

        // Process blue team
        if let Some(team_data) = blue_team_data {
            if let Some(ref mut conn) = self.blue_connection {
                if conn.state() == ConnectionState::Ready {
                    match conn.update(team_data, self.blue_skill_statuses.clone()) {
                        Ok(Some(strategy_output)) => {
                            output.blue_commands = strategy_output.skill_commands;
                            output.blue_roles = strategy_output.player_roles;
                            self.forward_debug_data(TeamColor::Blue, strategy_output.debug_data);
                        }
                        Ok(None) => {
                            // No response, strategy might be slow
                        }
                        Err(ConnectionError::Closed) => {
                            warn!("Blue team strategy connection closed");
                            self.blue_connection = None;
                        }
                        Err(e) => {
                            error!("Blue team strategy error: {}", e);
                        }
                    }
                }
            }
        }

        // Process yellow team
        if let Some(team_data) = yellow_team_data {
            if let Some(ref mut conn) = self.yellow_connection {
                if conn.state() == ConnectionState::Ready {
                    match conn.update(team_data, self.yellow_skill_statuses.clone()) {
                        Ok(Some(strategy_output)) => {
                            output.yellow_commands = strategy_output.skill_commands;
                            output.yellow_roles = strategy_output.player_roles;
                            self.forward_debug_data(TeamColor::Yellow, strategy_output.debug_data);
                        }
                        Ok(None) => {
                            // No response
                        }
                        Err(ConnectionError::Closed) => {
                            warn!("Yellow team strategy connection closed");
                            self.yellow_connection = None;
                        }
                        Err(e) => {
                            error!("Yellow team strategy error: {}", e);
                        }
                    }
                }
            }
        }

        output
    }

    /// Forward debug data to the dies-core debug system.
    fn forward_debug_data(&self, team: TeamColor, entries: Vec<DebugEntry>) {
        let prefix = match team {
            TeamColor::Blue => "strategy_blue",
            TeamColor::Yellow => "strategy_yellow",
        };

        for entry in entries {
            let key = format!("{}.{}", prefix, entry.key);
            let value = self.convert_debug_value(entry.value);
            debug_record(key, value);
        }
    }

    /// Convert protocol debug value to dies-core debug value.
    fn convert_debug_value(&self, value: dies_strategy_protocol::DebugValue) -> DebugValue {
        match value {
            dies_strategy_protocol::DebugValue::Shape(shape) => {
                DebugValue::Shape(self.convert_debug_shape(shape))
            }
            dies_strategy_protocol::DebugValue::Number(n) => DebugValue::Number(n),
            dies_strategy_protocol::DebugValue::String(s) => DebugValue::String(s),
        }
    }

    /// Convert protocol debug shape to dies-core debug shape.
    fn convert_debug_shape(&self, shape: dies_strategy_protocol::DebugShape) -> DebugShape {
        match shape {
            dies_strategy_protocol::DebugShape::Cross { center, color } => DebugShape::Cross {
                center,
                color: color.into(),
            },
            dies_strategy_protocol::DebugShape::Line { start, end, color } => DebugShape::Line {
                start,
                end,
                color: color.into(),
            },
            dies_strategy_protocol::DebugShape::Circle {
                center,
                radius,
                fill,
                stroke,
            } => DebugShape::Circle {
                center,
                radius,
                fill: fill.map(|c| c.into()),
                stroke: stroke.map(|c| c.into()),
            },
        }
    }

    /// Check if a team has an active strategy.
    pub fn has_strategy(&self, team: TeamColor) -> bool {
        match team {
            TeamColor::Blue => self.blue_connection.is_some(),
            TeamColor::Yellow => self.yellow_connection.is_some(),
        }
    }

    /// Check if a team's strategy is connected and ready.
    pub fn is_strategy_ready(&self, team: TeamColor) -> bool {
        match team {
            TeamColor::Blue => self
                .blue_connection
                .as_ref()
                .map(|c| c.state() == ConnectionState::Ready)
                .unwrap_or(false),
            TeamColor::Yellow => self
                .yellow_connection
                .as_ref()
                .map(|c| c.state() == ConnectionState::Ready)
                .unwrap_or(false),
        }
    }

    /// Get the name of the strategy running for a team.
    pub fn strategy_name(&self, team: TeamColor) -> Option<&str> {
        match team {
            TeamColor::Blue => self.config.blue_strategy.as_deref(),
            TeamColor::Yellow => self.config.yellow_strategy.as_deref(),
        }
    }

    /// Shutdown all strategies.
    pub fn shutdown(&mut self) {
        self.stop_strategy(TeamColor::Blue);
        self.stop_strategy(TeamColor::Yellow);
    }
}

impl Drop for StrategyHost {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StrategyHostConfig::default();
        assert!(config.blue_strategy.is_none());
        assert!(config.yellow_strategy.is_none());
    }

    #[test]
    fn test_host_creation() {
        let config = StrategyHostConfig::default();
        let host = StrategyHost::new(config);
        assert!(!host.has_strategy(TeamColor::Blue));
        assert!(!host.has_strategy(TeamColor::Yellow));
    }
}

