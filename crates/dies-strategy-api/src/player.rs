//! Per-player control interface.
//!
//! The [`PlayerHandle`] provides control over a single player. Strategies access
//! handles through [`TeamContext::players()`] or [`TeamContext::player()`].
//!
//! # Skill API Design
//!
//! The skill API uses a hybrid approach:
//!
//! - **Continuous skills** (`go_to`, `dribble_to`, `handle_ball`): Call each frame, parameters update smoothly
//! - **Discrete skills** (`reflex_shoot`, `snatch`): Start once, return handles for monitoring

use crate::skill_builders::{
    DribbleBuilder, GoToBoundedBuilder, GoToBuilder, HandleBallParams, ReflexReceiveParams,
    ReflexShootParams, SkillHandle, SnatchParams,
};
use dies_core::Angle;
use dies_strategy_protocol::{
    AcquirePosition, BallAction, MotionBounds, PlayerId, PlayerState, SkillCommand, SkillStatus,
    Vector2,
};
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

    /// Move to a target position with aggressive direct-velocity control,
    /// constrained to a bounded region.
    ///
    /// Like [`go_to`](Self::go_to) but bypasses the path follower and planner for
    /// a snappy response, while a no-overshoot velocity envelope keeps the robot
    /// inside `bounds`. Built for the goalkeeper's guard arc. Call each frame.
    ///
    /// ```ignore
    /// let zone = MotionBounds::Arc(ArcZone { center, min_radius, max_radius, half_angle });
    /// player.go_to_bounded(target, zone).facing(ball_pos);
    /// ```
    pub fn go_to_bounded(
        &mut self,
        position: Vector2,
        bounds: MotionBounds,
    ) -> GoToBoundedBuilder<'_> {
        GoToBoundedBuilder::new(self, position, bounds)
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

    // ========== Unified Ball Handling ==========

    /// Acquire the ball (if not held) and perform a continuously-supplied terminal
    /// [`BallAction`] with it — the unified acquire + carry + shoot/strike skill.
    ///
    /// Call each frame; swapping the `action` (e.g. `Hold` → `Shoot` once the ball
    /// is secured) is a live update on the same skill, so the acquire→act
    /// transition keeps its capture-phase state instead of restarting cold.
    ///
    /// - `acquire`: which side of the ball to take it from during the acquire
    ///   sub-phase — [`AcquirePosition::Fastest`] (closest reachable side),
    ///   [`AcquirePosition::Default`] (bias toward an exit heading derived from
    ///   `action`), or [`AcquirePosition::Heading`] (bias toward an explicit
    ///   heading).
    ///
    /// # Completion
    /// - `SkillStatus::Succeeded` when a kick departs (`Shoot`/`Strike`) or a `Carry`
    ///   reaches its target. `Hold` never self-completes.
    /// - `SkillStatus::Failed` on an unrecoverable acquire/aim/kick problem or a
    ///   per-action timeout (all timeouts are owned by the skill).
    pub fn handle_ball(
        &mut self,
        action: BallAction,
        acquire: AcquirePosition,
    ) -> SkillHandle<HandleBallParams> {
        // Magnet capture is opt-out (default on); flip the `magnet` field on the
        // returned handle's params (e.g. via `update_with`) to force velocity-only
        // capture for a given invocation.
        self.pending_command = Some(SkillCommand::HandleBall {
            action,
            acquire,
            magnet: true,
        });
        SkillHandle::new(HandleBallParams {
            action,
            acquire,
            magnet: true,
        })
    }

    /// Acquire the ball, aim at `target`, then release it via the firmware reflex
    /// kick — a single double-touch-safe contact from possession. Convenience for
    /// `handle_ball(BallAction::Strike { target, acquire_first: true },
    /// AcquirePosition::Default)`; use it for the designated kicker on an attacking
    /// restart, where a smart-kick's command-timed fire risks a double touch.
    pub fn strike_from_possession(&mut self, target: Vector2) -> SkillHandle<HandleBallParams> {
        self.handle_ball(
            BallAction::Strike {
                target,
                acquire_first: true,
            },
            AcquirePosition::Default,
        )
    }

    /// Drive through ball with reflex kick armed
    pub fn strike_through(&mut self, target: Vector2) -> SkillHandle<HandleBallParams> {
        self.handle_ball(
            BallAction::Strike {
                target,
                acquire_first: false,
            },
            AcquirePosition::Fastest,
        )
    }

    // ========== Discrete Skills ==========

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

    /// Receive a pass as a **one-timer**: position to intercept with the firmware
    /// reflex kick pre-armed and the robot facing `target`, so the kicker fires the
    /// instant the ball trips the breakbeam. The ball is never held — it is fired
    /// onward toward `target`.
    ///
    /// This is a **discrete skill** - start once and wait for completion. Returns a
    /// handle for monitoring and updating parameters.
    ///
    /// # Parameters
    ///
    /// - `from_pos`: position the ball is being passed from
    /// - `intercept_pos`: planned intercept point on the passing line
    /// - `target`: shot target the robot faces and fires toward
    /// - `capture_limit`: maximum perpendicular distance the receiver slides to meet
    ///   the ball
    ///
    /// # Geometry
    ///
    /// The robot faces `target`, so the incoming pass must arrive within the
    /// kicker-mouth acceptance cone (roughly: the passer in front of the receiver
    /// relative to `target`). Positioning for a feasible angle is the strategy's
    /// responsibility.
    ///
    /// # Completion
    ///
    /// - `SkillStatus::Succeeded` when the ball departs at speed after arriving
    /// - `SkillStatus::Failed` on timeout (the ball never arrives)
    pub fn reflex_receive(
        &mut self,
        from_pos: Vector2,
        intercept_pos: Vector2,
        target: Vector2,
        capture_limit: f64,
    ) -> SkillHandle<ReflexReceiveParams> {
        self.pending_command = Some(SkillCommand::ReflexReceive {
            from_pos,
            intercept_pos,
            target,
            capture_limit,
        });
        SkillHandle::new(ReflexReceiveParams {
            from_pos,
            intercept_pos,
            target,
            capture_limit,
        })
    }

    /// Strip the ball off an opponent that is holding it on its dribbler.
    ///
    /// This is a **discrete skill** - start once and wait for completion. The
    /// robot presses its dribbler against the held ball (keeping a body standoff so
    /// it contacts the *ball*, not the opponent chassis) and slowly rotates in
    /// place to peel the ball out of the opponent's dribbler. This is a
    /// *strip-for-a-teammate*: it succeeds when the opponent loses the ball.
    ///
    /// # Parameters
    ///
    /// - `release_hint`: Optional point to knock the ball loose toward (a
    ///   supporting teammate or open space). `None` defaults toward midfield.
    ///
    /// # Completion
    ///
    /// - `SkillStatus::Succeeded` when the targeted opponent no longer holds the ball
    /// - `SkillStatus::Failed` on timeout or if no opponent is holding the ball
    pub fn snatch(&mut self, release_hint: Option<Vector2>) -> SkillHandle<SnatchParams> {
        self.pending_command = Some(SkillCommand::Snatch { release_hint });
        SkillHandle::new(SnatchParams { release_hint })
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
    fn test_handle_ball_command() {
        let mut player = make_test_player();

        let _handle = player.handle_ball(
            BallAction::Hold {
                heading: Angle::from_radians(0.5),
            },
            AcquirePosition::Default,
        );

        let cmd = player.take_pending_command();
        assert!(cmd.is_some());

        match cmd.unwrap() {
            SkillCommand::HandleBall { action, .. } => {
                assert!(
                    matches!(action, BallAction::Hold { heading } if (heading.radians() - 0.5).abs() < 1e-6)
                );
            }
            _ => panic!("Expected HandleBall command"),
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
