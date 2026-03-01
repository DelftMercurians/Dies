//! Skill execution engine for the strategy-controlled path.
//!
//! This module provides the infrastructure for executing skills commanded by
//! strategies through the IPC protocol. It manages skill lifecycle, parameter
//! updates, and status reporting.

use std::collections::HashMap;

use dies_core::{PlayerData, PlayerId, TeamData};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use super::{PlayerControlInput, TeamContext};

/// The result of a skill execution step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkillResult {
    /// Skill completed successfully.
    Success,
    /// Skill failed to complete.
    Failure,
}

/// Progress of a skill during execution.
#[derive(Debug)]
pub enum SkillProgress {
    /// Skill continues, apply this control input.
    Continue(PlayerControlInput),
    /// Skill completed with the given result.
    Done(SkillResult),
}

impl SkillProgress {
    /// Creates a new `SkillProgress` with a `Success` result.
    pub fn success() -> SkillProgress {
        SkillProgress::Done(SkillResult::Success)
    }

    /// Creates a new `SkillProgress` with a `Failure` result.
    pub fn failure() -> SkillProgress {
        SkillProgress::Done(SkillResult::Failure)
    }
}

/// Context provided to skills during execution.
pub struct SkillContext<'a> {
    /// The player executing the skill.
    pub player: &'a PlayerData,
    /// The current world state.
    pub world: &'a TeamData,
    /// Team context for debug output.
    pub team_context: &'a TeamContext,
    /// Debug key prefix for this player.
    pub debug_prefix: String,
}

/// A discriminant for skill types, used for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkillType {
    GoToPos,
    Dribble,
    PickupBall,
    ReflexShoot,
    Stop,
}

impl From<&SkillCommand> for SkillType {
    fn from(cmd: &SkillCommand) -> Self {
        match cmd {
            SkillCommand::GoToPos { .. } => SkillType::GoToPos,
            SkillCommand::Dribble { .. } => SkillType::Dribble,
            SkillCommand::PickupBall { .. } => SkillType::PickupBall,
            SkillCommand::ReflexShoot { .. } => SkillType::ReflexShoot,
            SkillCommand::Stop => SkillType::Stop,
        }
    }
}

/// Trait for executable skills in the executor.
///
/// Skills implementing this trait can be:
/// - Updated with new parameters while running
/// - Executed tick-by-tick
/// - Queried for their status
pub trait ExecutableSkill: Send {
    /// Get the type of this skill.
    fn skill_type(&self) -> SkillType;

    /// Update parameters while the skill is running.
    ///
    /// This is called when the same skill type is commanded again while
    /// already running. The skill should smoothly transition to the new
    /// parameters.
    fn update_params(&mut self, command: &SkillCommand);

    /// Execute one tick of the skill.
    ///
    /// Returns either a control input to apply or a completion status.
    fn tick(&mut self, ctx: SkillContext<'_>) -> SkillProgress;

    /// Get the current status of the skill.
    fn status(&self) -> SkillStatus;
}

/// Per-player skill execution state.
struct PlayerSkillState {
    /// The currently running skill, if any.
    current_skill: Option<Box<dyn ExecutableSkill>>,
    /// The status of the last skill execution.
    last_status: SkillStatus,
}

impl Default for PlayerSkillState {
    fn default() -> Self {
        Self {
            current_skill: None,
            last_status: SkillStatus::Idle,
        }
    }
}

/// Manages skill execution for all players.
///
/// The `SkillExecutor` handles:
/// - Creating skill instances from commands
/// - Updating parameters when the same skill type is re-commanded
/// - Switching skills when a different type is commanded
/// - Tracking skill status for reporting back to strategies
pub struct SkillExecutor {
    /// Per-player skill state.
    player_states: HashMap<PlayerId, PlayerSkillState>,
}

impl SkillExecutor {
    /// Create a new skill executor.
    pub fn new() -> Self {
        Self {
            player_states: HashMap::new(),
        }
    }

    /// Process a skill command for a player.
    ///
    /// Returns the control input to apply this frame.
    ///
    /// # Command Semantics
    ///
    /// | Incoming Command  | Current Skill    | Action                                   |
    /// |-------------------|------------------|------------------------------------------|
    /// | None              | Any              | Continue current skill with last params  |
    /// | Same skill type   | Running          | Update parameters on existing skill      |
    /// | Different type    | Running          | Interrupt current, start new skill       |
    /// | Any command       | Succeeded/Failed | Start new skill instance                 |
    /// | Stop              | Any              | Interrupt current, robot stops           |
    pub fn process_command(
        &mut self,
        player_id: PlayerId,
        command: Option<&SkillCommand>,
        ctx: SkillContext<'_>,
    ) -> PlayerControlInput {
        let state = self.player_states.entry(player_id).or_default();

        // Handle Stop command specially - always immediate
        if matches!(command, Some(SkillCommand::Stop)) {
            state.current_skill = None;
            state.last_status = SkillStatus::Idle;
            return PlayerControlInput::default();
        }

        // If no command, continue current skill if any
        let Some(cmd) = command else {
            return self.tick_current_skill(player_id, ctx);
        };

        let cmd_type = SkillType::from(cmd);

        // Check if we need to start a new skill or update existing
        let need_new_skill = match &state.current_skill {
            None => true,
            Some(skill) => {
                // Different skill type -> start new
                if skill.skill_type() != cmd_type {
                    true
                } else {
                    // Same type, but if skill completed, start new instance
                    matches!(
                        state.last_status,
                        SkillStatus::Succeeded | SkillStatus::Failed
                    )
                }
            }
        };

        if need_new_skill {
            // Create new skill instance
            let skill = create_skill_from_command(cmd);
            state.current_skill = Some(skill);
            state.last_status = SkillStatus::Running;
        } else if let Some(skill) = &mut state.current_skill {
            // Update parameters on existing skill
            skill.update_params(cmd);
        }

        // Execute the skill
        self.tick_current_skill(player_id, ctx)
    }

    /// Tick the current skill for a player.
    fn tick_current_skill(
        &mut self,
        player_id: PlayerId,
        ctx: SkillContext<'_>,
    ) -> PlayerControlInput {
        let state = self.player_states.entry(player_id).or_default();

        let Some(skill) = &mut state.current_skill else {
            // No skill running, return idle input
            return PlayerControlInput::default();
        };

        // If skill already completed, don't tick it again
        if matches!(
            state.last_status,
            SkillStatus::Succeeded | SkillStatus::Failed
        ) {
            return PlayerControlInput::default();
        }

        // Execute the skill
        match skill.tick(ctx) {
            SkillProgress::Continue(input) => {
                state.last_status = SkillStatus::Running;
                input
            }
            SkillProgress::Done(result) => {
                state.last_status = match result {
                    SkillResult::Success => SkillStatus::Succeeded,
                    SkillResult::Failure => SkillStatus::Failed,
                };
                // Robot stops when skill completes
                PlayerControlInput::default()
            }
        }
    }

    /// Get the current skill status for a player.
    pub fn get_status(&self, player_id: PlayerId) -> SkillStatus {
        self.player_states
            .get(&player_id)
            .map(|s| s.last_status)
            .unwrap_or(SkillStatus::Idle)
    }

    /// Get the skill statuses for all players.
    pub fn get_all_statuses(&self) -> HashMap<PlayerId, SkillStatus> {
        self.player_states
            .iter()
            .map(|(id, state)| (*id, state.last_status))
            .collect()
    }

    /// Clear skill state for a player (e.g., when player is removed).
    pub fn clear_player(&mut self, player_id: PlayerId) {
        self.player_states.remove(&player_id);
    }
}

impl Default for SkillExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Create a skill instance from a command.
fn create_skill_from_command(cmd: &SkillCommand) -> Box<dyn ExecutableSkill> {
    use super::super::skills::executable::*;

    match cmd {
        SkillCommand::GoToPos { position, heading } => {
            Box::new(GoToPosSkill::new(*position, *heading))
        }
        SkillCommand::Dribble {
            target_pos,
            target_heading,
        } => Box::new(DribbleSkill::new(*target_pos, *target_heading)),
        SkillCommand::PickupBall { target_heading } => {
            Box::new(PickupBallSkill::new(*target_heading))
        }
        SkillCommand::ReflexShoot { target } => Box::new(ReflexShootSkill::new(*target)),
        SkillCommand::Stop => {
            // Stop is handled specially, should never reach here
            unreachable!("Stop command should be handled before create_skill_from_command")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{Angle, Vector2};

    #[test]
    fn test_skill_type_from_command() {
        let cmd = SkillCommand::GoToPos {
            position: Vector2::new(0.0, 0.0),
            heading: None,
        };
        assert_eq!(SkillType::from(&cmd), SkillType::GoToPos);

        let cmd = SkillCommand::Dribble {
            target_pos: Vector2::new(0.0, 0.0),
            target_heading: Angle::from_radians(0.0),
        };
        assert_eq!(SkillType::from(&cmd), SkillType::Dribble);

        let cmd = SkillCommand::PickupBall {
            target_heading: Angle::from_radians(0.0),
        };
        assert_eq!(SkillType::from(&cmd), SkillType::PickupBall);

        let cmd = SkillCommand::ReflexShoot {
            target: Vector2::new(0.0, 0.0),
        };
        assert_eq!(SkillType::from(&cmd), SkillType::ReflexShoot);

        assert_eq!(SkillType::from(&SkillCommand::Stop), SkillType::Stop);
    }

    #[test]
    fn test_executor_default_status() {
        let executor = SkillExecutor::new();
        assert_eq!(executor.get_status(PlayerId::new(1)), SkillStatus::Idle);
    }
}
