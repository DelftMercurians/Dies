//! IPC message types for host-strategy communication.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{DebugEntry, PlayerId, SkillCommand, SkillStatus, WorldSnapshot};

/// Configuration passed to a strategy on initialization.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[derive(Default)]
pub struct StrategyConfig {
    /// Strategy-specific configuration as a JSON string.
    /// Strategies can parse this however they need.
    pub custom_config: Option<String>,
}


/// Messages sent from the executor (host) to strategy processes.
///
/// Note: The `Init` message does **not** include team color - strategies don't
/// need to know their team color because all coordinates are pre-normalized.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HostMessage {
    /// Initial configuration when connecting.
    ///
    /// Sent once after the connection is established.
    Init {
        /// Strategy configuration.
        config: StrategyConfig,
    },

    /// World state update sent every frame.
    ///
    /// All coordinates are pre-transformed to the team-relative frame.
    WorldUpdate {
        /// Current world state snapshot.
        world: WorldSnapshot,
        /// Current skill status for each player.
        skill_statuses: HashMap<PlayerId, SkillStatus>,
    },

    /// Request graceful shutdown.
    ///
    /// Strategy should clean up and exit.
    Shutdown,
}

/// Log level for strategy log messages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Messages sent from strategy processes to the executor (host).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StrategyMessage {
    /// Indicates the strategy is ready to receive world updates.
    ///
    /// Sent once after receiving the `Init` message.
    Ready,

    /// Response to a world update.
    ///
    /// Sent after processing each `WorldUpdate` message.
    Output {
        /// Skill commands for each player.
        ///
        /// - `Some(command)`: Execute this skill command
        /// - `None`: Continue the current skill with previous parameters
        ///
        /// Players not included in this map will continue their current skill.
        skill_commands: HashMap<PlayerId, Option<SkillCommand>>,

        /// Debug visualization data to display in the UI.
        debug_data: Vec<DebugEntry>,

        /// Role names for each player (for UI display and debugging).
        player_roles: HashMap<PlayerId, String>,
    },

    /// Log message to forward to the host logging system.
    Log { level: LogLevel, message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SkillCommand;

    #[test]
    fn test_host_message_serialization() {
        // Test Init message
        let init = HostMessage::Init {
            config: StrategyConfig::default(),
        };
        let encoded = bincode::serialize(&init).unwrap();
        let decoded: HostMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            HostMessage::Init { config } => {
                assert!(config.custom_config.is_none());
            }
            _ => panic!("Expected Init message"),
        }

        // Test Shutdown message
        let shutdown = HostMessage::Shutdown;
        let encoded = bincode::serialize(&shutdown).unwrap();
        let decoded: HostMessage = bincode::deserialize(&encoded).unwrap();
        assert!(matches!(decoded, HostMessage::Shutdown));
    }

    #[test]
    fn test_strategy_message_serialization() {
        // Test Ready message
        let ready = StrategyMessage::Ready;
        let encoded = bincode::serialize(&ready).unwrap();
        let decoded: StrategyMessage = bincode::deserialize(&encoded).unwrap();
        assert!(matches!(decoded, StrategyMessage::Ready));

        // Test Output message
        let mut skill_commands = HashMap::new();
        skill_commands.insert(
            PlayerId::new(1),
            Some(SkillCommand::GoToPos {
                position: crate::Vector2::new(100.0, 200.0),
                heading: None,
            }),
        );
        skill_commands.insert(PlayerId::new(2), None);

        let mut player_roles = HashMap::new();
        player_roles.insert(PlayerId::new(1), "Striker".to_string());

        let output = StrategyMessage::Output {
            skill_commands,
            debug_data: vec![],
            player_roles,
        };

        let encoded = bincode::serialize(&output).unwrap();
        let decoded: StrategyMessage = bincode::deserialize(&encoded).unwrap();

        match decoded {
            StrategyMessage::Output {
                skill_commands,
                player_roles,
                ..
            } => {
                assert!(skill_commands.contains_key(&PlayerId::new(1)));
                assert!(skill_commands.contains_key(&PlayerId::new(2)));
                assert_eq!(player_roles.get(&PlayerId::new(1)), Some(&"Striker".to_string()));
            }
            _ => panic!("Expected Output message"),
        }

        // Test Log message
        let log = StrategyMessage::Log {
            level: LogLevel::Info,
            message: "Test log message".to_string(),
        };
        let encoded = bincode::serialize(&log).unwrap();
        let decoded: StrategyMessage = bincode::deserialize(&encoded).unwrap();
        match decoded {
            StrategyMessage::Log { level, message } => {
                assert_eq!(level, LogLevel::Info);
                assert_eq!(message, "Test log message");
            }
            _ => panic!("Expected Log message"),
        }
    }
}

