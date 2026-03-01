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
    /// - `target_heading`: Desired heading after ball is captured (for follow-up action)
    ///
    /// **Behavior**:
    /// 1. Move to position behind ball, facing `target_heading`
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
    ReflexShoot { target: Vector2 },

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_command_go_to_pos_serialization() {
        let cmd = SkillCommand::GoToPos {
            position: Vector2::new(1000.0, -500.0),
            heading: Some(Angle::from_radians(std::f64::consts::PI / 4.0)),
        };

        let encoded = bincode::serialize(&cmd).unwrap();
        let decoded: SkillCommand = bincode::deserialize(&encoded).unwrap();

        match decoded {
            SkillCommand::GoToPos { position, heading } => {
                assert!((position.x - 1000.0).abs() < 1e-6);
                assert!((position.y - (-500.0)).abs() < 1e-6);
                assert!(heading.is_some());
            }
            _ => panic!("Expected GoToPos"),
        }
    }

    #[test]
    fn test_skill_command_dribble_serialization() {
        let cmd = SkillCommand::Dribble {
            target_pos: Vector2::new(2000.0, 0.0),
            target_heading: Angle::from_radians(0.0),
        };

        let encoded = bincode::serialize(&cmd).unwrap();
        let decoded: SkillCommand = bincode::deserialize(&encoded).unwrap();

        match decoded {
            SkillCommand::Dribble {
                target_pos,
                target_heading,
            } => {
                assert!((target_pos.x - 2000.0).abs() < 1e-6);
                assert!((target_heading.radians() - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected Dribble"),
        }
    }

    #[test]
    fn test_skill_command_pickup_ball_serialization() {
        let cmd = SkillCommand::PickupBall {
            target_heading: Angle::from_radians(std::f64::consts::PI),
        };

        let encoded = bincode::serialize(&cmd).unwrap();
        let decoded: SkillCommand = bincode::deserialize(&encoded).unwrap();

        match decoded {
            SkillCommand::PickupBall { target_heading } => {
                assert!((target_heading.radians() - std::f64::consts::PI).abs() < 1e-6);
            }
            _ => panic!("Expected PickupBall"),
        }
    }

    #[test]
    fn test_skill_command_reflex_shoot_serialization() {
        let cmd = SkillCommand::ReflexShoot {
            target: Vector2::new(4500.0, 0.0),
        };

        let encoded = bincode::serialize(&cmd).unwrap();
        let decoded: SkillCommand = bincode::deserialize(&encoded).unwrap();

        match decoded {
            SkillCommand::ReflexShoot { target } => {
                assert!((target.x - 4500.0).abs() < 1e-6);
                assert!((target.y - 0.0).abs() < 1e-6);
            }
            _ => panic!("Expected ReflexShoot"),
        }
    }

    #[test]
    fn test_skill_command_stop_serialization() {
        let cmd = SkillCommand::Stop;

        let encoded = bincode::serialize(&cmd).unwrap();
        let decoded: SkillCommand = bincode::deserialize(&encoded).unwrap();

        assert!(matches!(decoded, SkillCommand::Stop));
    }

    #[test]
    fn test_skill_status_serialization() {
        for status in [
            SkillStatus::Idle,
            SkillStatus::Running,
            SkillStatus::Succeeded,
            SkillStatus::Failed,
        ] {
            let encoded = bincode::serialize(&status).unwrap();
            let decoded: SkillStatus = bincode::deserialize(&encoded).unwrap();
            assert_eq!(decoded, status);
        }
    }

    #[test]
    fn test_skill_status_default() {
        assert_eq!(SkillStatus::default(), SkillStatus::Idle);
    }
}
