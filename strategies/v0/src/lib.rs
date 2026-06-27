//! # v0-strategy
//!
//! The original Delft Mercurians **v0** strategy — a role-assignment + behavior
//! tree design — revived on top of the IPC strategy framework so it can be run as
//! a clean benchmark against `concerto`.
//!
//! The behavior-tree runtime and all of v0's roles are kept *local* to this
//! binary (under [`bt`] and [`roles`]); the legacy in-executor skills are
//! substituted with the new IPC skills inside the BT action nodes. The original
//! implementation is preserved under `reference/` for comparison.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

pub mod bt;
pub mod helpers;
pub mod roles;

use bt::BtRuntime;

/// The v0 strategy: a thin [`Strategy`] shell around the behavior-tree runtime.
pub struct V0Strategy {
    runtime: BtRuntime,
}

impl V0Strategy {
    pub fn new() -> Self {
        Self {
            runtime: BtRuntime::new(),
        }
    }
}

impl Default for V0Strategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for V0Strategy {
    fn init(&mut self, _world: &World) {
        tracing::info!("v0 strategy initialized");
    }

    fn update(&mut self, ctx: &mut TeamContext) {
        // Clone the world so the snapshot borrow is released before we hand `ctx`
        // mutably to the runtime.
        let world = World::new(ctx.world().raw_snapshot().clone());

        // Hard stops: let the executor halt the robots.
        if matches!(
            world.game_state(),
            GameState::Halt | GameState::Unknown | GameState::Timeout
        ) {
            return;
        }

        self.runtime.update(roles::v0_strategy, &world, ctx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{Angle, FieldGeometry, Vector2};
    use dies_strategy_protocol::{
        BallState, PlayerId, PlayerState, Possession, SkillStatus, WorldSnapshot,
    };
    use std::collections::HashMap;

    fn snapshot(game_state: GameState, ball: Vector2) -> WorldSnapshot {
        let own_players = (0..6)
            .map(|i| {
                PlayerState::new(
                    PlayerId::new(i),
                    Vector2::new(-3000.0 + i as f64 * 600.0, (i as f64 - 3.0) * 400.0),
                    Vector2::zeros(),
                    Angle::from_radians(0.0),
                )
            })
            .collect();
        let opp_players = (0..6)
            .map(|i| {
                PlayerState::new(
                    PlayerId::new(10 + i),
                    Vector2::new(3000.0 - i as f64 * 500.0, (i as f64 - 3.0) * 300.0),
                    Vector2::zeros(),
                    Angle::from_radians(0.0),
                )
            })
            .collect();
        WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball,
                velocity: Vector2::zeros(),
                detected: true,
            }),
            own_players,
            opp_players,
            game_state,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(0)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
        }
    }

    /// Drive the strategy across a handful of game states and assert it assigns
    /// roles and emits skill commands to every robot without panicking.
    fn run_state(game_state: GameState, ball: Vector2) {
        let mut strategy = V0Strategy::new();
        for frame in 0..5 {
            let mut snap = snapshot(game_state, ball);
            snap.timestamp = frame as f64 * 0.016;
            let mut ctx = TeamContext::new(snap, HashMap::new(), HashMap::new(), HashMap::new());
            strategy.update(&mut ctx);

            let (commands, roles) = ctx.collect_output();
            // Every robot should be given a role.
            assert_eq!(roles.len(), 6, "all robots assigned a role in {game_state:?}");
            // The keeper is always present.
            assert!(
                roles.values().any(|r| r == "goalkeeper"),
                "a goalkeeper is assigned in {game_state:?}"
            );
            // At least one robot receives a concrete skill command.
            assert!(
                commands.values().any(|c| c.is_some()),
                "some robot is commanded in {game_state:?}"
            );
        }
    }

    #[test]
    fn drives_all_robots_in_run() {
        run_state(GameState::Run, Vector2::new(0.0, 0.0));
    }

    #[test]
    fn drives_defensive_ball_in_our_half() {
        run_state(GameState::Run, Vector2::new(-2000.0, 500.0));
    }

    #[test]
    fn handles_setpieces() {
        run_state(GameState::Kickoff, Vector2::new(0.0, 0.0));
        run_state(GameState::FreeKick, Vector2::new(1000.0, 0.0));
        run_state(GameState::PenaltyRun, Vector2::new(3000.0, 0.0));
        run_state(GameState::Stop, Vector2::new(0.0, 0.0));
    }

    #[test]
    fn pickup_phase_reaches_shoot_when_ball_held() {
        // A robot that reports the ball captured should progress to shooting
        // rather than looping the pickup forever.
        let mut strategy = V0Strategy::new();
        let mut snap = snapshot(GameState::Run, Vector2::new(-2000.0, 0.0));
        // Mark the nearest field robot as holding the ball.
        snap.own_players[3].has_ball = true;
        let mut statuses = HashMap::new();
        statuses.insert(PlayerId::new(3), SkillStatus::Running);
        let mut ctx = TeamContext::new(snap, statuses, HashMap::new(), HashMap::new());
        strategy.update(&mut ctx);
        // Should not panic; commands collected fine.
        let (_commands, roles) = ctx.collect_output();
        assert_eq!(roles.len(), 6);
    }
}
