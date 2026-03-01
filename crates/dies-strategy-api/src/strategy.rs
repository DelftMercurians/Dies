//! Strategy trait definition.
//!
//! Strategies implement the [`Strategy`] trait to receive world updates and
//! control players through skill commands.

use crate::team::TeamContext;
use crate::world::World;

/// The interface that strategy implementations must implement.
///
/// Strategies are created once when loaded and receive updates each frame.
/// The `update()` method is called with the current world state and player handles.
///
/// # Lifecycle
///
/// 1. Strategy is constructed (via factory function)
/// 2. `init()` is called once with initial world state
/// 3. `update()` is called each frame
/// 4. `shutdown()` is called before unloading (optional)
///
/// # Example
///
/// ```ignore
/// use dies_strategy_api::prelude::*;
///
/// pub struct MyStrategy {
///     // State that persists across frames
///     target_positions: HashMap<PlayerId, Vector2>,
/// }
///
/// impl Strategy for MyStrategy {
///     fn init(&mut self, world: &World) {
///         // Initialize strategy state
///     }
///
///     fn update(&mut self, ctx: &mut TeamContext) {
///         let world = ctx.world();
///         let ball = world.ball_position();
///
///         for player in ctx.players() {
///             if let Some(ball_pos) = ball {
///                 player.go_to(ball_pos).facing(ball_pos);
///             }
///             player.set_role("Chaser");
///         }
///     }
/// }
///
/// // Export the strategy
/// dies_strategy_api::export_strategy!(MyStrategy::new);
/// ```
pub trait Strategy: Send + 'static {
    /// Called once when the strategy is loaded.
    ///
    /// Use this to initialize any state that depends on the world state,
    /// such as caching field geometry or initializing player-specific data.
    ///
    /// The default implementation does nothing.
    fn init(&mut self, _world: &World) {}

    /// Called each frame to update player skills.
    ///
    /// This is the main strategy logic. Use the `ctx` to:
    /// - Read world state via `ctx.world()`
    /// - Access player handles via `ctx.players()` or `ctx.player(id)`
    /// - Issue skill commands via player handles
    /// - Set player roles for visualization
    fn update(&mut self, ctx: &mut TeamContext);

    /// Called when the strategy is about to be unloaded.
    ///
    /// Use this for cleanup if needed. The default implementation does nothing.
    fn shutdown(&mut self) {}
}

/// Type alias for strategy factory functions.
///
/// Factory functions create new strategy instances. They take no arguments
/// and return a boxed strategy trait object.
pub type StrategyFactory = fn() -> Box<dyn Strategy>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use dies_strategy_protocol::{GameState, PlayerState, WorldSnapshot};
    use dies_core::Angle;

    struct TestStrategy {
        init_called: bool,
        update_count: u32,
        shutdown_called: bool,
    }

    impl TestStrategy {
        fn new() -> Self {
            Self {
                init_called: false,
                update_count: 0,
                shutdown_called: false,
            }
        }
    }

    impl Strategy for TestStrategy {
        fn init(&mut self, _world: &World) {
            self.init_called = true;
        }

        fn update(&mut self, ctx: &mut TeamContext) {
            self.update_count += 1;
            
            // Issue commands to all players
            for player in ctx.players() {
                player.go_to(dies_strategy_protocol::Vector2::new(0.0, 0.0));
            }
        }

        fn shutdown(&mut self) {
            self.shutdown_called = true;
        }
    }

    fn make_test_context() -> TeamContext {
        let snapshot = WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: None,
            ball: None,
            own_players: vec![PlayerState::new(
                dies_strategy_protocol::PlayerId::new(1),
                dies_strategy_protocol::Vector2::new(0.0, 0.0),
                dies_strategy_protocol::Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            )],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        };
        TeamContext::new(snapshot, HashMap::new())
    }

    #[test]
    fn test_strategy_lifecycle() {
        let mut strategy = TestStrategy::new();
        
        // Should start uninitialized
        assert!(!strategy.init_called);
        assert_eq!(strategy.update_count, 0);
        assert!(!strategy.shutdown_called);
        
        // Call init
        let world = World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.0,
            field_geom: None,
            ball: None,
            own_players: vec![],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        });
        strategy.init(&world);
        assert!(strategy.init_called);
        
        // Call update
        let mut ctx = make_test_context();
        strategy.update(&mut ctx);
        assert_eq!(strategy.update_count, 1);
        
        strategy.update(&mut ctx);
        assert_eq!(strategy.update_count, 2);
        
        // Call shutdown
        strategy.shutdown();
        assert!(strategy.shutdown_called);
    }

    #[test]
    fn test_default_implementations() {
        struct MinimalStrategy;
        
        impl Strategy for MinimalStrategy {
            fn update(&mut self, _ctx: &mut TeamContext) {
                // Do nothing
            }
        }
        
        let mut strategy = MinimalStrategy;
        
        // Default init and shutdown should not panic
        let world = World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.0,
            field_geom: None,
            ball: None,
            own_players: vec![],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        });
        strategy.init(&world);
        strategy.shutdown();
    }
}

