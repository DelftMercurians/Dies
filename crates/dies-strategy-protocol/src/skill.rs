//! Skill command and status types.
//!
//! Skills are the primary way strategies control robots. Each skill has specific
//! parameters and completion conditions.

use serde::{Deserialize, Serialize};

use crate::{Angle, Vector2};

/// A command to execute a skill on a player.
///
/// ## Skill Categories
///
/// Skills are divided into two categories:
///
/// ### Continuous Skills
/// - `GoToPos`, `Dribble`: Call each frame, parameters update smoothly
/// - If the same skill type is already running, parameters are updated without interruption
///
/// ### Discrete Skills
/// - `PickupBall`, `ReflexShoot`: Start once, run to completion
/// - Monitor via `SkillStatus`; can update parameters while running
///
/// ## Update Semantics
///
/// | Incoming Command  | Current Skill    | Action                                   |
/// |-------------------|------------------|------------------------------------------|
/// | None              | Any              | Continue current skill with last params  |
/// | Same skill type   | Running          | Update parameters on existing skill      |
/// | Different type    | Running          | Interrupt current, start new skill       |
/// | Any command       | Succeeded/Failed | Start new skill instance                 |
/// | Stop              | Any              | Interrupt current, robot stops           |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SkillCommand {
    /// Move to a target position.
    ///
    /// **Type**: Continuous - call each frame with updated position for smooth motion.
    ///
    /// **Parameters**:
    /// - `position`: Target position in mm
    /// - `heading`: Optional target heading; if `None`, robot maintains current heading
    ///
    /// **Completion**: `Succeeded` when robot arrives at position (within tolerance)
    GoToPos {
        position: Vector2,
        heading: Option<Angle>,
    },

    /// Dribble with the ball to a target position.
    ///
    /// **Type**: Continuous - call each frame with updated position.
    ///
    /// **Parameters**:
    /// - `target_pos`: Target position in mm
    /// - `target_heading`: Heading to maintain while dribbling
    ///
    /// **Behavior**:
    /// - Dribbler is activated
    /// - Uses limited acceleration to avoid losing the ball
    ///
    /// **Completion**:
    /// - `Succeeded` when robot arrives at position with ball
    /// - `Failed` if ball is lost during dribble
    Dribble {
        target_pos: Vector2,
        target_heading: Angle,
    },

    /// Approach and capture the ball.
    ///
    /// **Type**: Discrete - start once, wait for completion.
    ///
    /// **Parameters**:
    /// - `target_heading`: Desired heading after ball is captured. For a
    ///   stationary ball this also determines the approach direction —
    ///   the robot moves to the side of the ball opposite `target_heading`
    ///   so it ends up facing that direction at capture.
    ///
    /// **Behavior**:
    /// 1. Move to position behind ball (opposite side of `target_heading`),
    ///    already facing `target_heading`
    /// 2. Slowly approach ball with dribbler on
    /// 3. Complete when breakbeam detects ball
    ///
    /// **Completion**:
    /// - `Succeeded` when breakbeam detects ball
    /// - `Failed` if ball moves away or timeout
    PickupBall { target_heading: Angle },

    /// Orient toward a target and kick.
    ///
    /// **Type**: Discrete - start once, wait for completion.
    ///
    /// **Parameters**:
    /// - `target`: Position to kick toward (e.g., goal center)
    ///
    /// **Behavior**:
    /// 1. Rotate to face target
    /// 2. Arm kicker
    /// 3. Kick when aligned and ready
    ///
    /// **Completion**:
    /// - `Succeeded` when ball is kicked
    /// - `Failed` if angle is impossible or ball lost
    Shoot { target: Vector2 },

    /// Receive a pass by intercepting the ball along a passing line.
    ///
    /// **Type**: Discrete - start once, wait for completion.
    ///
    /// **Parameters**:
    /// - `from_pos`: Position the ball is being passed from
    /// - `target_pos`: Target position on the passing line where the receiver waits
    /// - `capture_limit`: Maximum distance the receiver moves perpendicular to the line
    /// - `cushion`: Whether to move with the ball direction to cushion the impact
    ///
    /// **Completion**:
    /// - `Succeeded` when breakbeam detects the ball
    Receive {
        from_pos: Vector2,
        target_pos: Vector2,
        capture_limit: f64,
        cushion: bool,
    },

    /// Stop all motion immediately.
    ///
    /// **Type**: Immediate
    ///
    /// **Behavior**:
    /// - Sets velocity to zero
    /// - Disables dribbler
    /// - Transitions to `Idle` status
    Stop,
}

/// The status of a skill's execution.
///
/// This is reported back to strategies each frame so they can monitor skill progress.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SkillStatus {
    /// No skill has been commanded yet for this player.
    #[default]
    Idle,

    /// The skill is currently executing.
    ///
    /// For continuous skills, this means the robot is moving toward the target.
    /// For discrete skills, this means the skill is in progress.
    Running,

    /// The skill completed successfully.
    ///
    /// - For `GoToPos`/`Dribble`: Robot arrived at target position
    /// - For `PickupBall`: Ball is captured (breakbeam triggered)
    /// - For `ReflexShoot`: Ball was kicked
    ///
    /// The robot is stopped. Status remains `Succeeded` until a new skill is commanded.
    Succeeded,

    /// The skill failed.
    ///
    /// - For `Dribble`: Ball was lost during dribble
    /// - For `PickupBall`: Ball moved away or timeout
    /// - For `ReflexShoot`: Could not align or ball lost
    ///
    /// The robot is stopped. Status remains `Failed` until a new skill is commanded.
    Failed,
}
