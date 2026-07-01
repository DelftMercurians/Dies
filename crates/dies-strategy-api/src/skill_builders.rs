//! Skill builders for ergonomic skill command construction.
//!
//! This module provides builder patterns for skill commands, making the API
//! more ergonomic while still generating the correct `SkillCommand` values.

use crate::player::PlayerHandle;
use dies_core::Angle;
use dies_strategy_protocol::{AcquirePosition, BallAction, MotionBounds, SkillCommand, Vector2};

/// Builder for `GoToPos` skill commands.
///
/// Created by [`PlayerHandle::go_to()`]. Commits the command when dropped.
///
/// # Example
///
/// ```ignore
/// // Simple: just move to position
/// player.go_to(target_pos);
///
/// // With heading
/// player.go_to(target_pos).with_heading(angle);
///
/// // Face toward a point
/// player.go_to(target_pos).facing(ball_pos);
/// ```
pub struct GoToBuilder<'a> {
    player: &'a mut PlayerHandle,
    position: Vector2,
    heading: Option<Angle>,
}

impl<'a> GoToBuilder<'a> {
    /// Create a new GoToBuilder.
    pub(crate) fn new(player: &'a mut PlayerHandle, position: Vector2) -> Self {
        Self {
            player,
            position,
            heading: None,
        }
    }

    /// Set the target heading.
    ///
    /// If not set, the robot maintains its current heading.
    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.heading = Some(heading);
        self
    }

    /// Compute and set heading to face the given position.
    ///
    /// This is a convenience method that calculates the angle from the target
    /// position to the given point.
    pub fn facing(mut self, look_at: Vector2) -> Self {
        let direction = look_at - self.position;
        if direction.norm() > 1e-6 {
            self.heading = Some(Angle::from_radians(direction.y.atan2(direction.x)));
        }
        self
    }
}

impl Drop for GoToBuilder<'_> {
    fn drop(&mut self) {
        self.player.set_pending_command(SkillCommand::GoToPos {
            position: self.position,
            heading: self.heading,
        });
    }
}

/// Builder for `GoToBounded` skill commands.
///
/// Created by [`PlayerHandle::go_to_bounded()`]. Commits the command when dropped.
///
/// # Example
///
/// ```ignore
/// player.go_to_bounded(target_pos, zone).facing(ball_pos);
/// ```
pub struct GoToBoundedBuilder<'a> {
    player: &'a mut PlayerHandle,
    position: Vector2,
    bounds: MotionBounds,
    heading: Option<Angle>,
}

impl<'a> GoToBoundedBuilder<'a> {
    /// Create a new GoToBoundedBuilder.
    pub(crate) fn new(
        player: &'a mut PlayerHandle,
        position: Vector2,
        bounds: MotionBounds,
    ) -> Self {
        Self {
            player,
            position,
            bounds,
            heading: None,
        }
    }

    /// Set the target heading. If not set, the robot maintains its current heading.
    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.heading = Some(heading);
        self
    }

    /// Compute and set heading to face the given position.
    pub fn facing(mut self, look_at: Vector2) -> Self {
        let direction = look_at - self.position;
        if direction.norm() > 1e-6 {
            self.heading = Some(Angle::from_radians(direction.y.atan2(direction.x)));
        }
        self
    }
}

impl Drop for GoToBoundedBuilder<'_> {
    fn drop(&mut self) {
        self.player.set_pending_command(SkillCommand::GoToBounded {
            position: self.position,
            heading: self.heading,
            bounds: self.bounds,
        });
    }
}

/// Builder for `Dribble` skill commands.
///
/// Created by [`PlayerHandle::dribble_to()`]. Commits the command when dropped.
///
/// # Example
///
/// ```ignore
/// player.dribble_to(target_pos, target_heading);
/// ```
pub struct DribbleBuilder<'a> {
    player: &'a mut PlayerHandle,
    target_pos: Vector2,
    target_heading: Angle,
}

impl<'a> DribbleBuilder<'a> {
    /// Create a new DribbleBuilder.
    pub(crate) fn new(
        player: &'a mut PlayerHandle,
        target_pos: Vector2,
        target_heading: Angle,
    ) -> Self {
        Self {
            player,
            target_pos,
            target_heading,
        }
    }

    /// Update the target position.
    pub fn to(mut self, position: Vector2) -> Self {
        self.target_pos = position;
        self
    }

    /// Update the target heading.
    pub fn with_heading(mut self, heading: Angle) -> Self {
        self.target_heading = heading;
        self
    }
}

impl Drop for DribbleBuilder<'_> {
    fn drop(&mut self) {
        self.player.set_pending_command(SkillCommand::Dribble {
            target_pos: self.target_pos,
            target_heading: self.target_heading,
        });
    }
}

/// Trait for skill parameter types.
pub trait SkillParams: Clone {
    /// Get the skill command for these parameters.
    fn to_command(&self) -> SkillCommand;
}

/// Parameters for `ReflexShoot` skill.
#[derive(Clone, Debug)]
pub struct ReflexShootParams {
    /// Target position to shoot toward.
    pub target: Vector2,
}

impl SkillParams for ReflexShootParams {
    fn to_command(&self) -> SkillCommand {
        SkillCommand::Shoot {
            target: self.target,
        }
    }
}

/// Parameters for the unified `HandleBall` skill.
#[derive(Clone, Copy, Debug)]
pub struct HandleBallParams {
    /// Terminal action to perform once the ball is held.
    pub action: BallAction,
    /// Which side of the ball to take it from during the acquire sub-phase.
    pub acquire: AcquirePosition,
    /// Allow firmware magnet capture during the commit drive (default `true`).
    /// Set `false` to force the velocity-only capture for this invocation.
    pub magnet: bool,
}

impl SkillParams for HandleBallParams {
    fn to_command(&self) -> SkillCommand {
        SkillCommand::HandleBall {
            action: self.action,
            acquire: self.acquire,
            magnet: self.magnet,
        }
    }
}

/// Parameters for the `ReflexReceive` one-timer skill.
#[derive(Clone, Debug)]
pub struct ReflexReceiveParams {
    /// Position the ball is being passed from.
    pub from_pos: Vector2,
    /// Planned intercept point on the passing line.
    pub intercept_pos: Vector2,
    /// Shot target the robot faces and fires the one-timer toward.
    pub target: Vector2,
    /// Maximum perpendicular distance the receiver slides to meet the ball.
    pub capture_limit: f64,
}

impl SkillParams for ReflexReceiveParams {
    fn to_command(&self) -> SkillCommand {
        SkillCommand::ReflexReceive {
            from_pos: self.from_pos,
            intercept_pos: self.intercept_pos,
            target: self.target,
            capture_limit: self.capture_limit,
        }
    }
}

/// Parameters for `Snatch` skill.
#[derive(Clone, Debug)]
pub struct SnatchParams {
    /// Optional point to knock the ball loose toward (e.g. a teammate or open
    /// space). `None` lets the skill default toward midfield.
    pub release_hint: Option<Vector2>,
}

impl SkillParams for SnatchParams {
    fn to_command(&self) -> SkillCommand {
        SkillCommand::Snatch {
            release_hint: self.release_hint,
        }
    }
}

/// Handle to a running discrete skill.
///
/// Provides methods to update skill parameters while the skill is running.
/// Dropping the handle does **not** cancel the skill.
///
/// # Example
///
/// ```ignore
/// // Start skill
/// let mut handle = player.handle_ball(BallAction::Hold { heading }, AcquirePosition::Default);
///
/// // Update parameters
/// handle.update_with(|p| p.action = BallAction::Shoot { target });
/// ```
#[derive(Clone, Debug)]
pub struct SkillHandle<S: SkillParams> {
    params: S,
}

impl<S: SkillParams> SkillHandle<S> {
    /// Create a new skill handle with initial parameters.
    pub(crate) fn new(params: S) -> Self {
        Self { params }
    }

    /// Get the current parameters.
    pub fn params(&self) -> &S {
        &self.params
    }

    /// Update the skill parameters.
    ///
    /// Note: The updated parameters won't be sent until the next frame when
    /// the strategy calls the skill method again.
    pub fn update(&mut self, params: S) {
        self.params = params;
    }

    /// Update parameters using a closure.
    ///
    /// Note: The updated parameters won't be sent until the next frame when
    /// the strategy calls the skill method again.
    pub fn update_with(&mut self, f: impl FnOnce(&mut S)) {
        f(&mut self.params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{PlayerId, PlayerState, SkillStatus};

    fn make_test_player() -> PlayerHandle {
        PlayerHandle::new(
            PlayerState::new(
                PlayerId::new(1),
                Vector2::new(0.0, 0.0),
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            ),
            SkillStatus::Idle,
        )
    }

    #[test]
    fn test_go_to_builder_basic() {
        let mut player = make_test_player();

        // Create and drop builder
        {
            let _ = GoToBuilder::new(&mut player, Vector2::new(100.0, 200.0));
        }

        let cmd = player.take_pending_command().unwrap();
        match cmd {
            SkillCommand::GoToPos { position, heading } => {
                assert_eq!(position.x, 100.0);
                assert_eq!(position.y, 200.0);
                assert!(heading.is_none());
            }
            _ => panic!("Expected GoToPos"),
        }
    }

    #[test]
    fn test_go_to_builder_with_heading() {
        let mut player = make_test_player();

        {
            let _ = GoToBuilder::new(&mut player, Vector2::new(100.0, 200.0))
                .with_heading(Angle::from_radians(1.5));
        }

        let cmd = player.take_pending_command().unwrap();
        match cmd {
            SkillCommand::GoToPos { heading, .. } => {
                assert!((heading.unwrap().radians() - 1.5).abs() < 1e-6);
            }
            _ => panic!("Expected GoToPos"),
        }
    }

    #[test]
    fn test_go_to_builder_facing() {
        let mut player = make_test_player();

        {
            // Position at (100, 0), facing (200, 0) -> heading should be 0
            let _ = GoToBuilder::new(&mut player, Vector2::new(100.0, 0.0))
                .facing(Vector2::new(200.0, 0.0));
        }

        let cmd = player.take_pending_command().unwrap();
        match cmd {
            SkillCommand::GoToPos { heading, .. } => {
                assert!(heading.unwrap().radians().abs() < 1e-6);
            }
            _ => panic!("Expected GoToPos"),
        }
    }

    #[test]
    fn test_skill_handle_update() {
        let mut handle = SkillHandle::new(HandleBallParams {
            action: BallAction::Hold {
                heading: Angle::from_radians(0.0),
            },
            acquire: AcquirePosition::Default,
            magnet: true,
        });

        handle.update_with(|p| p.acquire = AcquirePosition::Heading(Angle::from_radians(1.0)));

        assert!(matches!(
            handle.params().acquire,
            AcquirePosition::Heading(a) if (a.radians() - 1.0).abs() < 1e-6
        ));
    }
}
