//! # dies-strategy-api
//!
//! Public API for implementing dies strategies.
//!
//! This crate provides the interface that strategy implementations link against.
//! Strategies are compiled as separate binaries and communicate with the executor
//! via IPC.
//!
//! ## Overview
//!
//! Strategies implement the [`Strategy`] trait, which receives world state updates
//! and controls players through a skill-based API.
//!
//! ### Key Types
//!
//! - [`Strategy`]: The trait strategies implement
//! - [`TeamContext`]: Access to world state and player handles
//! - [`World`]: Read-only world state (normalized coordinates)
//! - [`PlayerHandle`]: Per-player control interface
//!
//! ## Coordinate System
//!
//! All coordinates are in a **team-relative frame**:
//! - **+x axis**: Points toward the opponent's goal (attacking direction)
//! - **-x axis**: Points toward our own goal (defending direction)
//!
//! Strategies never see absolute world coordinates or team color.
//!
//! ## Quick Start
//!
//! ```ignore
//! use dies_strategy_api::prelude::*;
//!
//! pub struct MyStrategy {
//!     // Strategy state
//! }
//!
//! impl MyStrategy {
//!     pub fn new() -> Self {
//!         Self {}
//!     }
//! }
//!
//! impl Strategy for MyStrategy {
//!     fn update(&mut self, ctx: &mut TeamContext) {
//!         let world = ctx.world();
//!         let ball = world.ball_position();
//!
//!         for player in ctx.players() {
//!             if let Some(ball_pos) = ball {
//!                 player.go_to(ball_pos).facing(ball_pos);
//!             }
//!             player.set_role("Chaser");
//!         }
//!     }
//! }
//!
//! // Export the strategy
//! dies_strategy_api::export_strategy!(MyStrategy::new);
//! ```
//!
//! ## Skill API
//!
//! The skill API uses a hybrid approach:
//!
//! ### Continuous Skills
//!
//! Call each frame - parameters update smoothly without interrupting the skill.
//!
//! - [`PlayerHandle::go_to`]: Move to a position with optional heading
//! - [`PlayerHandle::dribble_to`]: Move with ball, dribbler on
//!
//! ```ignore
//! // Simple movement
//! player.go_to(target_pos);
//!
//! // With heading control
//! player.go_to(target_pos).with_heading(angle);
//!
//! // Face toward a point
//! player.go_to(target_pos).facing(ball_pos);
//! ```
//!
//! ### Discrete Skills
//!
//! Start once and monitor status. Return handles for parameter updates.
//!
//! - [`PlayerHandle::pickup_ball`]: Approach and capture ball
//! - [`PlayerHandle::reflex_shoot`]: Orient and kick toward target
//!
//! ```ignore
//! // Start pickup
//! let handle = player.pickup_ball(target_heading);
//!
//! // Check status
//! match player.skill_status() {
//!     SkillStatus::Running => { /* still working */ }
//!     SkillStatus::Succeeded => { /* got ball */ }
//!     SkillStatus::Failed => { /* missed */ }
//!     _ => {}
//! }
//! ```
//!
//! ## Debug Visualization
//!
//! The [`debug`] module provides functions for adding visualizations to the UI.
//!
//! ```ignore
//! use dies_strategy_api::debug;
//!
//! debug::cross("target", target_pos);
//! debug::line("path", start, end);
//! debug::circle("zone", center, radius);
//! debug::value("speed", velocity.norm());
//! ```

pub mod debug;
mod player;
mod skill_builders;
mod strategy;
mod team;
mod world;

pub use player::PlayerHandle;
pub use skill_builders::{DribbleBuilder, GoToBuilder, PickupBallParams, ReflexShootParams, SkillHandle, SkillParams};
pub use strategy::{Strategy, StrategyFactory};
pub use team::TeamContext;
pub use world::{Rect, World};

// Re-export commonly used types from protocol and core crates
pub use dies_core::{Angle, FieldGeometry};
pub use dies_strategy_protocol::{
    BallState, DebugColor, DebugEntry, DebugShape, DebugValue, GameState, Handicap,
    PlayerId, PlayerState, SkillCommand, SkillStatus, Vector2, WorldSnapshot,
};

/// Prelude module for convenient imports.
///
/// Import everything commonly needed with:
///
/// ```ignore
/// use dies_strategy_api::prelude::*;
/// ```
pub mod prelude {
    pub use crate::debug;
    pub use crate::player::PlayerHandle;
    pub use crate::skill_builders::{PickupBallParams, ReflexShootParams, SkillHandle};
    pub use crate::strategy::Strategy;
    pub use crate::team::TeamContext;
    pub use crate::world::{Rect, World};

    // Core types
    pub use dies_core::{Angle, FieldGeometry};
    pub use dies_strategy_protocol::{
        BallState, DebugColor, GameState, Handicap, PlayerId, PlayerState, SkillCommand,
        SkillStatus, Vector2,
    };
}

/// Export a strategy factory function.
///
/// This macro generates the necessary boilerplate to export a strategy from
/// a strategy crate. The factory function is called when the strategy is loaded.
///
/// # Usage
///
/// ```ignore
/// use dies_strategy_api::prelude::*;
///
/// pub struct MyStrategy { /* ... */ }
///
/// impl MyStrategy {
///     pub fn new() -> Self {
///         Self { /* ... */ }
///     }
/// }
///
/// impl Strategy for MyStrategy {
///     fn update(&mut self, ctx: &mut TeamContext) {
///         // ...
///     }
/// }
///
/// // Export the strategy with the factory function
/// dies_strategy_api::export_strategy!(MyStrategy::new);
/// ```
///
/// # Generated Code
///
/// This macro generates:
/// 1. A `create_strategy` function that returns `Box<dyn Strategy>`
/// 2. A `STRATEGY_FACTORY` static that can be used by the runner
#[macro_export]
macro_rules! export_strategy {
    ($factory:expr) => {
        /// Create a new instance of the strategy.
        #[no_mangle]
        pub fn create_strategy() -> Box<dyn $crate::Strategy> {
            Box::new($factory())
        }

        /// The strategy factory function.
        #[no_mangle]
        pub static STRATEGY_FACTORY: fn() -> Box<dyn $crate::Strategy> = || {
            Box::new($factory())
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct TestStrategy;

    impl TestStrategy {
        fn new() -> Self {
            Self
        }
    }

    impl Strategy for TestStrategy {
        fn update(&mut self, _ctx: &mut TeamContext) {}
    }

    #[test]
    fn test_prelude_imports() {
        // Verify that prelude provides expected types
        use super::prelude::*;

        let _: PlayerId = PlayerId::new(1);
        let _: Vector2 = Vector2::new(0.0, 0.0);
        let _: Angle = Angle::from_radians(0.0);
        let _: GameState = GameState::Run;
        let _: SkillStatus = SkillStatus::Idle;
    }

    #[test]
    fn test_strategy_creation() {
        // Test that we can create a strategy and it implements the trait
        let mut strategy = TestStrategy::new();
        
        let snapshot = WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: None,
            ball: None,
            own_players: vec![],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        };

        let mut ctx = TeamContext::new(snapshot, HashMap::new());
        strategy.update(&mut ctx);
    }

    #[test]
    fn test_world_rect() {
        let rect = Rect::new(Vector2::new(0.0, 0.0), Vector2::new(100.0, 50.0));
        
        assert!(rect.contains(Vector2::new(50.0, 25.0)));
        assert!(!rect.contains(Vector2::new(150.0, 25.0)));
        assert_eq!(rect.width(), 100.0);
        assert_eq!(rect.height(), 50.0);
    }
}

