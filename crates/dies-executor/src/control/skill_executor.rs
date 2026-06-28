//! Skill execution engine for the strategy-controlled path.
//!
//! This module provides the infrastructure for executing skills commanded by
//! strategies through the IPC protocol. It manages skill lifecycle, parameter
//! updates, and status reporting.

use std::collections::HashMap;

use dies_core::{PlayerData, PlayerId, PlayerSkillInfo, SkillState, TeamData};
use dies_strategy_protocol::{SkillCommand, SkillStatus};

use super::avoidance::ObstacleSet;
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
    /// Static geometry + frozen opponent discs this robot should treat as
    /// blockers when choosing ball-relative poses (capture approach points, shot
    /// launch points). Excludes the ball itself — capture/aim skills drive *to*
    /// it. Empty when the caller has no obstacle context (e.g. the pass FSM's
    /// secure pickup). Opponents are a static snapshot; ORCA handles them
    /// reactively once the skill picks a target.
    pub obstacles: ObstacleSet,
}

/// Trait for executable skills in the executor.
///
/// Skills implementing this trait can be:
/// - Updated with new parameters while running
/// - Executed tick-by-tick
/// - Queried for their status
pub trait ExecutableSkill: Send {
    /// Return true if the given command matches this skill's type and can be used to update it.
    fn matches_command(&self, command: &SkillCommand) -> bool;

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

    /// Short, stable name for this skill type, shown in the UI (e.g. `"GoToPos"`).
    fn skill_type(&self) -> &'static str;

    /// Human-readable description of the skill's current internal state, shown in
    /// the UI alongside the status (e.g. `"approaching ball"`, `"aiming"`).
    fn description(&self) -> String;

    /// Whether this is a one-shot skill that must NOT be auto-restarted when the
    /// same command type keeps arriving after it has completed.
    ///
    /// Continuous skills (e.g. `GoToPos`, `Dribble`) return `false` (the default):
    /// a repeated command after completion re-activates them against the latest
    /// params. One-shot skills with an irreversible effect (e.g. shooting) return
    /// `true` so they latch their terminal result instead of re-firing — a fresh
    /// instance is only created when the commanded skill type actually changes.
    fn is_oneshot(&self) -> bool {
        false
    }
}

/// Map the executor's `SkillStatus` to the UI-facing `SkillState` in `dies-core`.
pub(crate) fn skill_state_from_status(status: SkillStatus) -> SkillState {
    match status {
        SkillStatus::Idle => SkillState::Idle,
        SkillStatus::Running => SkillState::Running,
        SkillStatus::Succeeded => SkillState::Succeeded,
        SkillStatus::Failed => SkillState::Failed,
    }
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

        // Check if we need to start a new skill or update existing
        let need_new_skill = match &state.current_skill {
            None => true,
            Some(skill) => {
                // Different skill type -> start new
                if !skill.matches_command(cmd) {
                    true
                } else {
                    // Same type, terminal: how it ended decides whether we retry.
                    // A *failed* skill never produced its irreversible effect, so
                    // re-creating and retrying is always safe — even a one-shot
                    // (e.g. a shot aborted before the kick fired). A *succeeded*
                    // one-shot latches so a repeated command can't re-fire it (the
                    // kick already happened); continuous skills re-activate.
                    match state.last_status {
                        SkillStatus::Failed => true,
                        SkillStatus::Succeeded => !skill.is_oneshot(),
                        _ => false,
                    }
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

    /// Get rich, UI-facing skill info for every player with an active skill.
    pub fn get_all_infos(&self) -> HashMap<PlayerId, PlayerSkillInfo> {
        self.player_states
            .iter()
            .filter_map(|(id, state)| {
                let skill = state.current_skill.as_ref()?;
                Some((
                    *id,
                    PlayerSkillInfo {
                        skill_type: skill.skill_type().to_string(),
                        state: skill_state_from_status(state.last_status),
                        description: skill.description(),
                    },
                ))
            })
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
        SkillCommand::GoToBounded {
            position,
            heading,
            bounds,
        } => Box::new(GoToBoundedSkill::new(*position, *heading, *bounds)),
        SkillCommand::Dribble {
            target_pos,
            target_heading,
        } => Box::new(DribbleSkill::new(*target_pos, *target_heading)),
        SkillCommand::PickupBall {
            target_heading,
            instant_kick,
        } => Box::new(PickupBallSkill::new(*target_heading, *instant_kick)),
        SkillCommand::Shoot { target } => Box::new(ShootSkill::new(*target)),
        SkillCommand::DribbleShoot { target } => Box::new(DribbleShootSkill::new(*target)),
        SkillCommand::HandleBall { action, approach } => {
            Box::new(HandleBallSkill::new(*action, *approach))
        }
        SkillCommand::Receive {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        } => Box::new(ReceiveSkill::new(
            *from_pos,
            *target_pos,
            *capture_limit,
            *cushion,
        )),
        SkillCommand::Snatch { release_hint } => Box::new(SnatchSkill::new(*release_hint)),
        SkillCommand::Pass { .. } => {
            // Pass is a joint skill handled by the JointSkillExecutor; the team
            // controller partitions it away from the per-player path.
            unreachable!("Pass command must be routed to the JointSkillExecutor")
        }
        SkillCommand::Stop => {
            // Stop is handled specially, should never reach here
            unreachable!("Stop command should be handled before create_skill_from_command")
        }
    }
}
