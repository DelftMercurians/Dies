//! Per-player control interface.
//!
//! The [`PlayerHandle`] provides control over a single player. Strategies access
//! handles through [`TeamContext::players()`] or [`TeamContext::player()`].
//!
//! # Skill API Design
//!
//! The skill API uses a hybrid approach:
//!
//! - **Continuous skills** (`go_to`, `dribble_to`): Call each frame, parameters update smoothly
//! - **Discrete skills** (`pickup_ball`, `reflex_shoot`): Start once, return handles for monitoring

use crate::skill_builders::{
    DribbleBuilder, GoToBuilder, PickupBallParams, ReceiveParams, ReflexShootParams, SkillHandle,
};
use dies_core::Angle;
use dies_strategy_protocol::{PlayerId, PlayerState, SkillCommand, SkillStatus, Vector2};
use std::collections::HashSet;

/// Per-player control interface.
///
/// Provides read access to player state and skill commands for controlling the player.
///
/// # Example
///
/// ```ignore
/// fn update(&mut self, ctx: &mut TeamContext) {
///     for player in ctx.players() {
///         // Read state
///         let pos = player.position();
///         let has_ball = player.has_ball();
///         
///         // Command skills
///         if has_ball {
///             player.dribble_to(goal, Angle::from_radians(0.0));
///         } else {
///             player.go_to(ball_pos).facing(ball_pos);
///         }
///         
///         player.set_role("Striker");
///     }
/// }
/// ```
pub struct PlayerHandle {
    /// Player state from the world snapshot.
    state: PlayerState,
    /// Current skill status as reported by the executor.
    skill_status: SkillStatus,
    /// Skill command to send this frame (None = continue previous).
    pending_command: Option<SkillCommand>,
    /// Role name for UI display.
    role: Option<String>,
}

impl PlayerHandle {
    /// Create a new PlayerHandle.
    pub(crate) fn new(state: PlayerState, skill_status: SkillStatus) -> Self {
        Self {
            state,
            skill_status,
            pending_command: None,
            role: None,
        }
    }

    // ========== Read-only State ==========

    /// Get the player's ID.
    pub fn id(&self) -> PlayerId {
        self.state.id
    }

    /// Get the player's current position in mm.
    pub fn position(&self) -> Vector2 {
        self.state.position
    }

    /// Get the player's current velocity in mm/s.
    pub fn velocity(&self) -> Vector2 {
        self.state.velocity
    }

    /// Get the player's current heading (yaw angle).
    pub fn heading(&self) -> Angle {
        self.state.heading
    }

    /// Get the player's angular velocity in rad/s.
    pub fn angular_velocity(&self) -> f64 {
        self.state.angular_velocity
    }

    /// Check if the player has the ball (breakbeam sensor triggered).
    pub fn has_ball(&self) -> bool {
        self.state.has_ball
    }

    /// Get the player's handicaps (hardware limitations).
    pub fn handicaps(&self) -> &HashSet<dies_strategy_protocol::Handicap> {
        &self.state.handicaps
    }

    /// Get the current skill execution status.
    pub fn skill_status(&self) -> SkillStatus {
        self.skill_status
    }

    /// Get the full player state (for advanced use cases).
    pub fn state(&self) -> &PlayerState {
        &self.state
    }

    // ========== Continuous Skills ==========

    /// Move to a target position.
    ///
    /// This is a **continuous skill** - call each frame with updated position for smooth motion.
    /// If the same skill type is already running, parameters are updated without interruption.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Simple movement
    /// player.go_to(target_pos);
    ///
    /// // With heading control
    /// player.go_to(target_pos).with_heading(Angle::from_radians(0.0));
    ///
    /// // Face toward a point
    /// player.go_to(target_pos).facing(ball_pos);
    /// ```
    ///
    /// # Completion
    ///
    /// `SkillStatus::Succeeded` when the robot arrives at the target position.
    pub fn go_to(&mut self, position: Vector2) -> GoToBuilder<'_> {
        GoToBuilder::new(self, position)
    }

    /// Dribble with the ball to a target position.
    ///
    /// This is a **continuous skill** - call each frame with updated position.
    ///
    /// # Behavior
    ///
    /// - Dribbler is activated
    /// - Uses limited acceleration to avoid losing the ball
    /// - Can rotate to align with target heading while moving
    ///
    /// # Completion
    ///
    /// - `SkillStatus::Succeeded` when robot arrives at position with ball
    /// - `SkillStatus::Failed` if ball is lost during dribble
    ///
    /// # Example
    ///
    /// ```ignore
    /// player.dribble_to(target_pos, Angle::from_radians(0.0));
    /// ```
    pub fn dribble_to(&mut self, position: Vector2, heading: Angle) -> DribbleBuilder<'_> {
        DribbleBuilder::new(self, position, heading)
    }

    // ========== Discrete Skills ==========

    /// Approach and capture the ball.
    ///
    /// This is a **discrete skill** - start once and wait for completion.
    /// Returns a handle for monitoring and updating parameters.
    ///
    /// # Parameters
    ///
    /// - `target_heading`: Desired heading after ball is captured (for follow-up action)
    ///
    /// # Behavior
    ///
    /// 1. Move to position behind ball, facing `target_heading`
    /// 2. Slowly approach ball with dribbler on
    /// 3. Complete when breakbeam detects ball
    ///
    /// # Completion
    ///
    /// - `SkillStatus::Succeeded` when breakbeam detects ball
    /// - `SkillStatus::Failed` if ball moves away or timeout
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Start pickup
    /// let handle = player.pickup_ball(heading_toward_goal);
    ///
    /// // Later, update heading while approaching
    /// handle.update_with(|p| p.target_heading = new_heading);
    /// ```
    pub fn pickup_ball(&mut self, target_heading: Angle) -> SkillHandle<PickupBallParams> {
        self.pending_command = Some(SkillCommand::PickupBall { target_heading });
        SkillHandle::new(PickupBallParams { target_heading })
    }

    /// Orient toward a target and kick.
    ///
    /// This is a **discrete skill** - start once and wait for completion.
    /// Returns a handle for monitoring.
    ///
    /// # Parameters
    ///
    /// - `target`: Position to kick toward (e.g., goal center)
    ///
    /// # Behavior
    ///
    /// 1. Rotate to face target
    /// 2. Arm kicker
    /// 3. Kick when aligned and ready
    ///
    /// # Completion
    ///
    /// - `SkillStatus::Succeeded` when ball is kicked
    /// - `SkillStatus::Failed` if angle is impossible or ball lost
    ///
    /// # Example
    ///
    /// ```ignore
    /// let handle = player.reflex_shoot(goal_center);
    ///
    /// // Check status
    /// if player.skill_status() == SkillStatus::Succeeded {
    ///     // Shot complete!
    /// }
    /// ```
    pub fn reflex_shoot(&mut self, target: Vector2) -> SkillHandle<ReflexShootParams> {
        self.pending_command = Some(SkillCommand::Shoot { target });
        SkillHandle::new(ReflexShootParams { target })
    }

    /// Receive a pass by intercepting along the passing line.
    ///
    /// This is a **discrete skill** - start once and wait for completion.
    /// Returns a handle for monitoring and updating parameters.
    ///
    /// # Parameters
    ///
    /// - `from_pos`: Position the ball is being passed from
    /// - `target_pos`: Target position on the passing line where the receiver waits
    /// - `capture_limit`: Maximum perpendicular distance the receiver moves to intercept
    /// - `cushion`: Whether to cushion the ball on impact
    pub fn receive(
        &mut self,
        from_pos: Vector2,
        target_pos: Vector2,
        capture_limit: f64,
        cushion: bool,
    ) -> SkillHandle<ReceiveParams> {
        self.pending_command = Some(SkillCommand::Receive {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        });
        SkillHandle::new(ReceiveParams {
            from_pos,
            target_pos,
            capture_limit,
            cushion,
        })
    }

    // ========== Control ==========

    /// Stop all motion immediately.
    ///
    /// - Sets velocity to zero
    /// - Disables dribbler
    /// - Transitions to `SkillStatus::Idle`
    pub fn stop(&mut self) {
        self.pending_command = Some(SkillCommand::Stop);
    }

    // ========== Metadata ==========

    /// Set the role name for debugging/visualization in the UI.
    ///
    /// This appears in the debug view next to the player.
    pub fn set_role(&mut self, role: &str) {
        self.role = Some(role.to_string());
    }

    /// Get the currently set role name.
    pub fn role(&self) -> Option<&str> {
        self.role.as_deref()
    }

    // ========== Internal ==========

    /// Set the pending command (used by skill builders).
    pub(crate) fn set_pending_command(&mut self, command: SkillCommand) {
        self.pending_command = Some(command);
    }

    /// Take the pending command (used by TeamContext to collect commands).
    pub(crate) fn take_pending_command(&mut self) -> Option<SkillCommand> {
        self.pending_command.take()
    }

    /// Take the role (used by TeamContext to collect roles).
    pub(crate) fn take_role(&mut self) -> Option<String> {
        self.role.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_player() -> PlayerHandle {
        PlayerHandle::new(
            PlayerState::new(
                PlayerId::new(1),
                Vector2::new(1000.0, 500.0),
                Vector2::new(100.0, 0.0),
                Angle::from_radians(0.5),
            )
            .with_has_ball(true)
            .with_angular_velocity(0.1),
            SkillStatus::Idle,
        )
    }

    #[test]
    fn test_player_state_access() {
        let player = make_test_player();

        assert_eq!(player.id(), PlayerId::new(1));
        assert_eq!(player.position(), Vector2::new(1000.0, 500.0));
        assert_eq!(player.velocity(), Vector2::new(100.0, 0.0));
        assert!((player.heading().radians() - 0.5).abs() < 1e-6);
        assert!(player.has_ball());
        assert_eq!(player.skill_status(), SkillStatus::Idle);
    }

    #[test]
    fn test_go_to_command() {
        let mut player = make_test_player();

        player.go_to(Vector2::new(2000.0, 1000.0));

        let cmd = player.take_pending_command();
        assert!(cmd.is_some());

        match cmd.unwrap() {
            SkillCommand::GoToPos { position, heading } => {
                assert_eq!(position, Vector2::new(2000.0, 1000.0));
                assert!(heading.is_none());
            }
            _ => panic!("Expected GoToPos command"),
        }
    }

    #[test]
    fn test_go_to_with_heading() {
        let mut player = make_test_player();

        player
            .go_to(Vector2::new(2000.0, 1000.0))
            .with_heading(Angle::from_radians(1.0));

        let cmd = player.take_pending_command();
        assert!(cmd.is_some());

        match cmd.unwrap() {
            SkillCommand::GoToPos { position, heading } => {
                assert_eq!(position, Vector2::new(2000.0, 1000.0));
                assert!(heading.is_some());
                assert!((heading.unwrap().radians() - 1.0).abs() < 1e-6);
            }
            _ => panic!("Expected GoToPos command"),
        }
    }

    #[test]
    fn test_stop_command() {
        let mut player = make_test_player();

        player.stop();

        let cmd = player.take_pending_command();
        assert!(matches!(cmd, Some(SkillCommand::Stop)));
    }

    #[test]
    fn test_role_setting() {
        let mut player = make_test_player();

        assert!(player.role().is_none());

        player.set_role("Striker");
        assert_eq!(player.role(), Some("Striker"));

        let role = player.take_role();
        assert_eq!(role, Some("Striker".to_string()));
        assert!(player.role().is_none());
    }

    #[test]
    fn test_pickup_ball_command() {
        let mut player = make_test_player();

        let _handle = player.pickup_ball(Angle::from_radians(0.5));

        let cmd = player.take_pending_command();
        assert!(cmd.is_some());

        match cmd.unwrap() {
            SkillCommand::PickupBall { target_heading } => {
                assert!((target_heading.radians() - 0.5).abs() < 1e-6);
            }
            _ => panic!("Expected PickupBall command"),
        }
    }

    #[test]
    fn test_reflex_shoot_command() {
        let mut player = make_test_player();

        let _handle = player.reflex_shoot(Vector2::new(4500.0, 0.0));

        let cmd = player.take_pending_command();
        assert!(cmd.is_some());

        match cmd.unwrap() {
            SkillCommand::Shoot { target } => {
                assert_eq!(target, Vector2::new(4500.0, 0.0));
            }
            _ => panic!("Expected ReflexShoot command"),
        }
    }
}
