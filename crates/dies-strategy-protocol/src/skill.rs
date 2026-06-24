//! Skill command and status types.
//!
//! Skills are the primary way strategies control robots. Each skill has specific
//! parameters and completion conditions.

use serde::{Deserialize, Serialize};

use crate::{Angle, PlayerId, Vector2};

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

    /// Aim by orbiting the captured ball, then kick.
    ///
    /// **Type**: Discrete - start once, wait for completion.
    ///
    /// **Parameters**:
    /// - `target_heading`: Direction to shoot. The robot slides tangentially
    ///   around the ball (dribbler engaged) until its shoot axis aligns with
    ///   this heading.
    ///
    /// **Completion**:
    /// - `Succeeded` when the ball is kicked
    /// - `Failed` if the ball is lost, the shoot lane is blocked, or timeout
    DribbleShoot { target_heading: Angle },

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

    /// Coordinate a pass between two players.
    ///
    /// **Type**: Joint - this command occupies the skill slot of BOTH the passer
    /// and the receiver. The same logical pass is commanded to each robot, with the
    /// other named as `partner`. A dedicated joint coordinator (ticked once per
    /// frame, not per-player) owns both robots through one state machine.
    ///
    /// **Parameters**:
    /// - `partner`: The other robot in the pass (receiver if this is the passer,
    ///   passer if this is the receiver)
    /// - `role`: Whether this robot is the `Passer` or `Receiver`
    /// - `target_hint`: Optional bias for where the receiver should end up; if
    ///   `None`, the coordinator computes the intercept geometry itself
    ///
    /// **Completion** (joint - both robots report the same status):
    /// - `Succeeded` when the receiver's breakbeam (or vision fallback) confirms possession
    /// - `Failed` with a typed [`PassResult`] reason otherwise (the coordinator
    ///   never leaves a robot stuck; both are released on any terminal outcome)
    Pass {
        partner: PlayerId,
        role: PassRole,
        target_hint: Option<Vector2>,
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

/// The role a robot plays in a [`SkillCommand::Pass`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassRole {
    /// Secures the ball, aims, and kicks toward the receiver.
    Passer,
    /// Moves to the intercept point and captures the ball.
    Receiver,
}

/// The rich, typed outcome of a pass.
///
/// Unlike the generic [`SkillStatus`] (which collapses to `Succeeded`/`Failed`),
/// this carries *why* a pass failed and where the ball ended up, so the strategy
/// can react intelligently. It is delivered to the strategy out-of-band via the
/// `pass_results` map on the world update.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PassResult {
    /// The receiver captured the ball.
    Success { receiver: PlayerId },
    /// The pass failed; see `reason`. Both robots have been released cleanly.
    Failure {
        reason: PassFailure,
        ball_state: PassBallState,
    },
}

/// Why a pass failed. The only unambiguous success is the receiver's breakbeam;
/// every other terminal condition is one of these.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PassFailure {
    /// The passer never secured the ball (not within secure distance, or lost it
    /// before the kick). The pass never chases a loose ball.
    BallLost,
    /// The ball was kicked but the receiver could not reach the intercept point
    /// (trajectory diverged beyond the capture window).
    ReceiverMissed,
    /// The ball was kicked but stopped short of the receiver.
    StoppedShort,
    /// An opponent intercepted the ball in flight.
    Intercepted,
    /// A phase exceeded its time budget (aim, receiver positioning, or flight).
    Timeout,
    /// The partner's slot stopped referencing this pass (reassigned/stopped by the
    /// strategy). The orphaned side releases cleanly.
    PartnerLeft,
    /// The strategy explicitly cancelled (e.g. `Stop`) before completion.
    Cancelled,
}

/// A minimal snapshot of where the ball ended up when a pass terminated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum PassBallState {
    /// The receiver has the ball.
    WithReceiver,
    /// The passer still has the ball (failed before/at kick).
    WithPasser,
    /// The ball is loose at the given position.
    Loose { position: Vector2 },
    /// An opponent has/controls the ball.
    WithOpponent,
    /// Possession could not be determined.
    Unknown,
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
