//! # Test Strategy
//!
//! A simple test strategy for validating the strategy pipeline.
//!
//! This strategy:
//! - Positions all robots in a formation
//! - Uses debug visualization to show target positions
//! - Validates that connection, world data, skill commands, and debug all work
//!
//! ## Formation
//!
//! The robots are positioned in a defensive arc formation near their own goal:
//! - Player 0 (if exists): Goalkeeper position at own goal
//! - Player 1: Left defender
//! - Player 2: Right defender
//! - Player 3: Left midfielder
//! - Player 4: Right midfielder
//! - Player 5+: Forward positions

use dies_strategy_api::debug;
use dies_strategy_api::prelude::*;

/// Test strategy that positions robots in a formation.
pub struct TestStrategy {
    /// Frame counter for animation/debugging.
    frame_count: u64,
    /// Whether init was called.
    initialized: bool,
    /// Field half length (cached from init).
    field_half_length: f64,
    /// Field half width (cached from init).
    field_half_width: f64,
}

impl TestStrategy {
    /// Create a new test strategy.
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            initialized: false,
            field_half_length: 4500.0, // Default SSL field
            field_half_width: 3000.0,
        }
    }

    /// Get the target position for a player based on their index in the team.
    fn get_formation_position(&self, player_index: usize) -> Vector2 {
        // All positions are relative to our goal (at -x)
        // +x direction is toward opponent goal

        let half_len = self.field_half_length;
        let half_width = self.field_half_width;

        match player_index {
            // Goalkeeper - in front of own goal
            0 => Vector2::new(-half_len + 500.0, 0.0),

            // Left defender
            1 => Vector2::new(-half_len + 1500.0, half_width * 0.4),

            // Right defender
            2 => Vector2::new(-half_len + 1500.0, -half_width * 0.4),

            // Left midfielder
            3 => Vector2::new(-half_len + 3000.0, half_width * 0.3),

            // Right midfielder
            4 => Vector2::new(-half_len + 3000.0, -half_width * 0.3),

            // Center forward
            5 => Vector2::new(0.0, 0.0),

            // Additional players - spread across midfield
            n => {
                let offset = ((n - 6) as f64) * 400.0;
                Vector2::new(-half_len + 2500.0, offset - 800.0)
            }
        }
    }

    /// Get the role name for a player based on their index.
    fn get_role_name(&self, player_index: usize) -> &'static str {
        match player_index {
            0 => "Goalkeeper",
            1 => "Left Defender",
            2 => "Right Defender",
            3 => "Left Mid",
            4 => "Right Mid",
            5 => "Striker",
            _ => "Reserve",
        }
    }
}

impl Default for TestStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for TestStrategy {
    fn init(&mut self, world: &World) {
        tracing::info!("TestStrategy initialized");

        // Cache field dimensions
        self.field_half_length = world.field_length() / 2.0;
        self.field_half_width = world.field_width() / 2.0;

        self.initialized = true;

        // Log the field dimensions
        debug::value("field.half_length", self.field_half_length);
        debug::value("field.half_width", self.field_half_width);
        debug::string("status", "initialized");
    }

    fn update(&mut self, ctx: &mut TeamContext) {
        self.frame_count += 1;

        // Update field dimensions if available (in case they weren't in init)
        if !self.initialized {
            self.field_half_length = ctx.world().field_length() / 2.0;
            self.field_half_width = ctx.world().field_width() / 2.0;
        }

        // Record debug info
        debug::value("frame_count", self.frame_count as f64);

        // Get ball position early (before borrowing ctx mutably)
        let ball_pos = ctx.world().ball_position();
        let own_goal = ctx.world().own_goal_center();
        let opp_goal = ctx.world().opp_goal_center();
        let player_count = ctx.player_count();

        if let Some(ball) = ball_pos {
            debug::cross("ball", ball);
            debug::value("ball.x", ball.x);
            debug::value("ball.y", ball.y);
        }

        debug::value("player_count", player_count as f64);

        // Get player IDs and collect them to avoid borrow conflicts
        let player_ids: Vec<PlayerId> = ctx.player_ids().to_vec();

        // Draw all target positions first
        for (i, _id) in player_ids.iter().enumerate() {
            let target = self.get_formation_position(i);
            let key = format!("target_{}", i);
            debug::cross_colored(&key, target, DebugColor::Blue);
        }

        // Now command each player
        for (i, id) in player_ids.iter().enumerate() {
            let target = self.get_formation_position(i);
            let role = self.get_role_name(i);

            if let Some(player) = ctx.player(*id) {
                // Draw line from current position to target
                let line_key = format!("path_{}", i);
                debug::line_colored(&line_key, player.position(), target, DebugColor::Green);

                // Move to formation position and face the ball (or center if no ball)
                if let Some(ball) = ball_pos {
                    player.go_to(target).facing(ball);
                } else {
                    // Face toward opponent goal (+x direction)
                    player.go_to(target).with_heading(Angle::from_radians(0.0));
                }

                // Set role for UI
                player.set_role(role);
            }
        }

        // Draw formation visualization
        debug::circle(
            "formation_arc",
            Vector2::new(-self.field_half_length + 2000.0, 0.0),
            1500.0,
        );

        // Draw goal areas
        debug::cross_colored("own_goal", own_goal, DebugColor::Yellow);
        debug::cross_colored("opp_goal", opp_goal, DebugColor::Red);

        // Log status periodically
        if self.frame_count % 100 == 0 {
            tracing::debug!(
                frame = self.frame_count,
                players = player_count,
                "TestStrategy update"
            );
        }
    }

    fn shutdown(&mut self) {
        tracing::info!(
            "TestStrategy shutting down after {} frames",
            self.frame_count
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{GameState, WorldSnapshot};
    use std::collections::HashMap;

    fn make_test_snapshot() -> WorldSnapshot {
        WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: None,
            ball: Some(BallState {
                position: Vector2::new(0.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: vec![
                dies_strategy_protocol::PlayerState::new(
                    PlayerId::new(0),
                    Vector2::new(-4000.0, 0.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
                dies_strategy_protocol::PlayerState::new(
                    PlayerId::new(1),
                    Vector2::new(-3000.0, 1000.0),
                    Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
            ],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(0)),
            freekick_kicker: None,
        }
    }

    #[test]
    fn test_strategy_creation() {
        let strategy = TestStrategy::new();
        assert!(!strategy.initialized);
        assert_eq!(strategy.frame_count, 0);
    }

    #[test]
    fn test_formation_positions() {
        let strategy = TestStrategy::new();

        // Goalkeeper should be near own goal
        let gk_pos = strategy.get_formation_position(0);
        assert!(gk_pos.x < -3000.0, "Goalkeeper should be near own goal");

        // Striker should be at center
        let striker_pos = strategy.get_formation_position(5);
        assert!(striker_pos.x.abs() < 100.0, "Striker should be near center");
    }

    #[test]
    fn test_update_issues_commands() {
        let mut strategy = TestStrategy::new();
        let snapshot = make_test_snapshot();
        let mut ctx = TeamContext::new(snapshot, HashMap::new());

        strategy.update(&mut ctx);

        let (commands, roles) = ctx.collect_output();

        // Both players should have commands
        assert!(commands.get(&PlayerId::new(0)).unwrap().is_some());
        assert!(commands.get(&PlayerId::new(1)).unwrap().is_some());

        // Both should have roles
        assert!(roles.get(&PlayerId::new(0)).is_some());
        assert!(roles.get(&PlayerId::new(1)).is_some());
    }

    #[test]
    fn test_role_names() {
        let strategy = TestStrategy::new();

        assert_eq!(strategy.get_role_name(0), "Goalkeeper");
        assert_eq!(strategy.get_role_name(1), "Left Defender");
        assert_eq!(strategy.get_role_name(5), "Striker");
        assert_eq!(strategy.get_role_name(10), "Reserve");
    }
}
