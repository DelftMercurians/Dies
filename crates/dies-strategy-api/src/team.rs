//! Team context providing access to world state and player handles.
//!
//! The [`TeamContext`] is the main interface passed to strategy `update()` methods.
//! It provides access to the world state and control over all own players.

use std::collections::HashMap;

use dies_strategy_protocol::{
    ParamValue, PassResult, PassRole, PlayerId, SkillCommand, SkillStatus, StrategyParams, Vector2,
    WorldSnapshot,
};

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
    pass_results: HashMap<PlayerId, PassResult>,
    params: StrategyParams,
}

impl TeamContext {
    /// Create a new TeamContext from a world snapshot, skill statuses, and pass
    /// results.
    pub fn new(
        snapshot: WorldSnapshot,
        skill_statuses: HashMap<PlayerId, SkillStatus>,
        pass_results: HashMap<PlayerId, PassResult>,
        params: StrategyParams,
    ) -> Self {
        let player_ids: Vec<PlayerId> = snapshot.own_players.iter().map(|p| p.id).collect();

        let players = snapshot
            .own_players
            .iter()
            .map(|p| {
                let status = skill_statuses
                    .get(&p.id)
                    .copied()
                    .unwrap_or(SkillStatus::Idle);
                (p.id, PlayerHandle::new(p.clone(), status))
            })
            .collect();

        Self {
            world: World::new(snapshot),
            players,
            player_ids,
            pass_results,
            params,
        }
    }

    /// Read a boolean runtime parameter (`false` if unset / not a bool).
    pub fn param_bool(&self, key: &str) -> bool {
        self.params
            .get(key)
            .and_then(ParamValue::as_bool)
            .unwrap_or(false)
    }

    /// Read a float runtime parameter (`0.0` if unset / not numeric).
    pub fn param_float(&self, key: &str) -> f64 {
        self.params
            .get(key)
            .and_then(ParamValue::as_float)
            .unwrap_or(0.0)
    }

    /// Read an integer runtime parameter (`0` if unset / not numeric).
    pub fn param_int(&self, key: &str) -> i32 {
        self.params
            .get(key)
            .and_then(ParamValue::as_int)
            .unwrap_or(0)
    }

    /// Read a text runtime parameter (empty string if unset / not text).
    pub fn param_text(&self, key: &str) -> String {
        self.params
            .get(key)
            .and_then(ParamValue::as_text)
            .unwrap_or("")
            .to_string()
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

    /// Command an atomic pass between two players.
    ///
    /// This is a *joint* action: the same pass is written into the skill slots of
    /// both the passer and the receiver, and a single executor-side coordinator
    /// drives both robots. Poll the joint outcome via either member's
    /// [`skill_status()`](PlayerHandle::skill_status) and read the typed reason via
    /// [`pass_result()`](Self::pass_result).
    ///
    /// Cancel it like any skill — command anything else on either robot.
    ///
    /// # Example
    /// ```ignore
    /// ctx.pass(passer_id, receiver_id);              // coordinator picks geometry
    /// ctx.pass(passer_id, receiver_id).target_hint(p); // bias the receive point
    /// ctx.pass(passer_id, receiver_id).forward_to(goal); // one-timer redirect
    /// ```
    pub fn pass(&mut self, passer: PlayerId, receiver: PlayerId) -> PassBuilder<'_> {
        PassBuilder {
            ctx: self,
            passer,
            receiver,
            target_hint: None,
            forward_to: None,
        }
    }

    /// The rich result for a player involved in a pass, if any.
    ///
    /// Present on (and retained after) the frame the pass terminates, for both
    /// members. Use this to branch on the typed [`PassFailure`] reason that the
    /// generic [`SkillStatus`] cannot carry.
    pub fn pass_result(&self, id: PlayerId) -> Option<&PassResult> {
        self.pass_results.get(&id)
    }

    /// Write a pass command into both members' slots (used by [`PassBuilder`]).
    fn commit_pass(
        &mut self,
        passer: PlayerId,
        receiver: PlayerId,
        target_hint: Option<Vector2>,
        forward_to: Option<Vector2>,
    ) {
        if let Some(h) = self.players.get_mut(&passer) {
            h.set_pending_command(SkillCommand::Pass {
                partner: receiver,
                role: PassRole::Passer,
                target_hint,
                forward_to,
            });
        }
        if let Some(h) = self.players.get_mut(&receiver) {
            h.set_pending_command(SkillCommand::Pass {
                partner: passer,
                role: PassRole::Receiver,
                target_hint,
                forward_to,
            });
        }
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
    pub fn collect_output(
        &mut self,
    ) -> (
        HashMap<PlayerId, Option<SkillCommand>>,
        HashMap<PlayerId, String>,
    ) {
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

/// Builder for [`TeamContext::pass`]. Commits the pass into both players' slots
/// when dropped (matching the commit-on-drop idiom of the continuous skills).
pub struct PassBuilder<'a> {
    ctx: &'a mut TeamContext,
    passer: PlayerId,
    receiver: PlayerId,
    target_hint: Option<Vector2>,
    forward_to: Option<Vector2>,
}

impl PassBuilder<'_> {
    /// Bias where the receiver should end up. If unset, the coordinator computes
    /// the intercept point from the receiver's position.
    pub fn target_hint(mut self, target: Vector2) -> Self {
        self.target_hint = Some(target);
        self
    }

    /// Make the pass a **one-timer**: the receiver waits with the reflex kick
    /// pre-armed, facing `target`, and redirects the arriving ball toward it the
    /// instant it trips the breakbeam — it never takes possession. Success then
    /// reports `forwarded: true` (or `false` if the reflex dudded and the ball
    /// stuck to the dribbler — still a completed pass).
    ///
    /// The coordinator trusts the caller's geometry: keep the deflection angle
    /// (incoming pass vs receiver→`target`) shallow, roughly ≲60°, or the ball
    /// glances off the shell. The angle is emitted as a debug value.
    pub fn forward_to(mut self, target: Vector2) -> Self {
        self.forward_to = Some(target);
        self
    }
}

impl Drop for PassBuilder<'_> {
    fn drop(&mut self) {
        self.ctx
            .commit_pass(self.passer, self.receiver, self.target_hint, self.forward_to);
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
            pre_stage: false,
            our_keeper_id: None,
            freekick_kicker: None,
            double_touch_barred: None,
            possession: dies_strategy_protocol::Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        }
    }

    #[test]
    fn test_team_context_creation() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();

        let ctx = TeamContext::new(snapshot, skill_statuses, HashMap::new(), HashMap::new());

        assert_eq!(ctx.player_count(), 2);
        assert_eq!(ctx.player_ids().len(), 2);
    }

    #[test]
    fn test_player_access() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();

        let mut ctx = TeamContext::new(snapshot, skill_statuses, HashMap::new(), HashMap::new());

        assert!(ctx.player(PlayerId::new(1)).is_some());
        assert!(ctx.player(PlayerId::new(2)).is_some());
        assert!(ctx.player(PlayerId::new(99)).is_none());
    }

    #[test]
    fn test_collect_output() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();

        let mut ctx = TeamContext::new(snapshot, skill_statuses, HashMap::new(), HashMap::new());

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

        let ctx = TeamContext::new(snapshot, skill_statuses, HashMap::new(), HashMap::new());

        let world = ctx.world();
        assert!(world.ball().is_some());
        assert_eq!(world.game_state(), GameState::Run);
    }

    #[test]
    fn test_param_accessors() {
        let snapshot = make_test_snapshot();
        let mut params = HashMap::new();
        params.insert("defense_only".to_string(), ParamValue::Bool(true));
        params.insert("aggression".to_string(), ParamValue::Float(0.7));

        let ctx = TeamContext::new(snapshot, HashMap::new(), HashMap::new(), params);

        // Set values are returned.
        assert!(ctx.param_bool("defense_only"));
        assert_eq!(ctx.param_float("aggression"), 0.7);
        // Unset keys fall back to zero-defaults.
        assert!(!ctx.param_bool("missing"));
        assert_eq!(ctx.param_float("missing"), 0.0);
        assert_eq!(ctx.param_int("missing"), 0);
        assert_eq!(ctx.param_text("missing"), "");
    }

    #[test]
    fn test_players_iterator() {
        let snapshot = make_test_snapshot();
        let skill_statuses = HashMap::new();

        let mut ctx = TeamContext::new(snapshot, skill_statuses, HashMap::new(), HashMap::new());

        let mut count = 0;
        for player in ctx.players() {
            player.set_role("Test");
            count += 1;
        }

        assert_eq!(count, 2);
    }
}
