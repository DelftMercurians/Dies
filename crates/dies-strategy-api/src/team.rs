//! Team context providing access to world state and player handles.
//!
//! The [`TeamContext`] is the main interface passed to strategy `update()` methods.
//! It provides access to the world state and control over all own players.

use std::collections::HashMap;

use dies_strategy_protocol::{PlayerId, SkillCommand, SkillStatus, WorldSnapshot};

use crate::player::PlayerHandle;
use crate::world::World;

/// Access to world state and all own player handles.
///
/// This is the main interface passed to strategy `update()` methods. It provides:
/// - Read-only access to world state via [`world()`](Self::world)
/// - Mutable access to player handles for issuing skill commands
///
/// # Example
///
/// ```ignore
/// impl Strategy for MyStrategy {
///     fn update(&mut self, ctx: &mut TeamContext) {
///         let world = ctx.world();
///         let ball_pos = world.ball_position();
///         
///         for player in ctx.players() {
///             player.go_to(ball_pos.unwrap_or_default());
///         }
///     }
/// }
/// ```
pub struct TeamContext {
    world: World,
    players: HashMap<PlayerId, PlayerHandle>,
    player_ids: Vec<PlayerId>,
}

impl TeamContext {
    /// Create a new TeamContext from a world snapshot and skill statuses.
    pub fn new(snapshot: WorldSnapshot, skill_statuses: HashMap<PlayerId, SkillStatus>) -> Self {
        let player_ids: Vec<PlayerId> = snapshot.own_players.iter().map(|p| p.id).collect();
        
        let players = snapshot
            .own_players
            .iter()
            .map(|p| {
                let status = skill_statuses.get(&p.id).copied().unwrap_or(SkillStatus::Idle);
                (p.id, PlayerHandle::new(p.clone(), status))
            })
            .collect();
        
        Self {
            world: World::new(snapshot),
            players,
            player_ids,
        }
    }

    /// Read-only access to world state.
    ///
    /// All coordinates are in the normalized team-relative frame where +x points
    /// toward the opponent's goal.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Iterate over all own player handles.
    ///
    /// # Example
    ///
    /// ```ignore
    /// for player in ctx.players() {
    ///     player.go_to(target_pos);
    ///     player.set_role("Support");
    /// }
    /// ```
    pub fn players(&mut self) -> impl Iterator<Item = &mut PlayerHandle> {
        self.players.values_mut()
    }

    /// Get a specific player handle by ID.
    ///
    /// Returns `None` if no player with that ID exists.
    pub fn player(&mut self, id: PlayerId) -> Option<&mut PlayerHandle> {
        self.players.get_mut(&id)
    }

    /// Get an immutable reference to a player handle by ID.
    pub fn player_ref(&self, id: PlayerId) -> Option<&PlayerHandle> {
        self.players.get(&id)
    }

    /// Get the list of own player IDs.
    ///
    /// This is useful for iterating over player IDs without borrowing the context mutably.
    pub fn player_ids(&self) -> &[PlayerId] {
        &self.player_ids
    }

    /// Get the number of own players.
    pub fn player_count(&self) -> usize {
        self.players.len()
    }

    /// Collect all pending skill commands and player roles.
    ///
    /// This is called by the runner after the strategy's `update()` method.
    /// Returns `(skill_commands, player_roles)`.
    pub fn collect_output(&mut self) -> (HashMap<PlayerId, Option<SkillCommand>>, HashMap<PlayerId, String>) {
        let mut skill_commands = HashMap::new();
        let mut player_roles = HashMap::new();
        
        for (id, player) in self.players.iter_mut() {
            let cmd = player.take_pending_command();
            skill_commands.insert(*id, cmd);
            
            if let Some(role) = player.take_role() {
                player_roles.insert(*id, role);
            }
        }
        
        (skill_commands, player_roles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::Angle;
    use dies_strategy_protocol::{BallState, GameState, PlayerState};

    fn make_test_snapshot() -> WorldSnapshot {
        WorldSnapshot {
            timestamp: 1.0,
            dt: 0.016,
            field_geom: None,
            ball: Some(BallState {
                position: dies_strategy_protocol::Vector2::new(0.0, 0.0),
                velocity: dies_strategy_protocol::Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: vec![
                PlayerState::new(
                    PlayerId::new(1),
                    dies_strategy_protocol::Vector2::new(1000.0, 500.0),
                    dies_strategy_protocol::Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
                PlayerState::new(
                    PlayerId::new(2),
                    dies_strategy_protocol::Vector2::new(-500.0, 0.0),
                    dies_strategy_protocol::Vector2::new(0.0, 0.0),
                    Angle::from_radians(0.0),
                ),
            ],
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: None,
            freekick_kicker: None,
        }
    }

    #[test]
    fn test_team_context_creation() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();
        
        let ctx = TeamContext::new(snapshot, skill_statuses);
        
        assert_eq!(ctx.player_count(), 2);
        assert_eq!(ctx.player_ids().len(), 2);
    }

    #[test]
    fn test_player_access() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();
        
        let mut ctx = TeamContext::new(snapshot, skill_statuses);
        
        assert!(ctx.player(PlayerId::new(1)).is_some());
        assert!(ctx.player(PlayerId::new(2)).is_some());
        assert!(ctx.player(PlayerId::new(99)).is_none());
    }

    #[test]
    fn test_collect_output() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();
        
        let mut ctx = TeamContext::new(snapshot, skill_statuses);
        
        // Issue commands
        if let Some(player) = ctx.player(PlayerId::new(1)) {
            player.go_to(dies_strategy_protocol::Vector2::new(100.0, 100.0));
            player.set_role("Striker");
        }
        
        let (commands, roles) = ctx.collect_output();
        
        // Player 1 should have a command
        assert!(commands.get(&PlayerId::new(1)).unwrap().is_some());
        // Player 2 should have None (no command issued)
        assert!(commands.get(&PlayerId::new(2)).unwrap().is_none());
        
        // Check roles
        assert_eq!(roles.get(&PlayerId::new(1)), Some(&"Striker".to_string()));
        assert!(roles.get(&PlayerId::new(2)).is_none());
    }

    #[test]
    fn test_world_access() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();
        
        let ctx = TeamContext::new(snapshot, skill_statuses);
        
        let world = ctx.world();
        assert!(world.ball().is_some());
        assert_eq!(world.game_state(), GameState::Run);
    }

    #[test]
    fn test_players_iterator() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();
        
        let mut ctx = TeamContext::new(snapshot, skill_statuses);
        
        let mut count = 0;
        for player in ctx.players() {
            player.set_role("Test");
            count += 1;
        }
        
        assert_eq!(count, 2);
    }
}

