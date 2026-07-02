//! Coordinate transformation between world and team-relative frames.
//!
//! Strategies operate in a normalized coordinate system where:
//! - +x always points toward the opponent's goal
//! - -x always points toward our own goal
//!
//! This module handles converting between the absolute world coordinates
//! and the team-relative coordinates that strategies use.

use std::collections::HashSet;

use dies_core::{Angle, PlayerData, SideAssignment, TeamColor, TeamData, Vector2};
use dies_strategy_protocol::{
    BallContest, BallState, DebugEntry, DebugShape, DebugValue, Handicap, PassBallState,
    PassResult, PlayerId, PlayerState, Possession, SkillCommand, SkillStatus, WorldSnapshot,
};

/// Handles coordinate transformation between world and strategy frames.
///
/// The transformer maintains the current side assignment and team color,
/// and provides methods to convert coordinates in both directions.
#[derive(Debug, Clone)]
pub struct CoordinateTransformer {
    team_color: TeamColor,
    side_assignment: SideAssignment,
}

impl CoordinateTransformer {
    /// Create a new coordinate transformer for a team.
    pub fn new(team_color: TeamColor, side_assignment: SideAssignment) -> Self {
        Self {
            team_color,
            side_assignment,
        }
    }

    /// Update the side assignment (e.g., at halftime).
    pub fn set_side_assignment(&mut self, side_assignment: SideAssignment) {
        self.side_assignment = side_assignment;
    }

    /// Get the direction sign for coordinate transformation.
    ///
    /// Returns 1.0 if coordinates don't need flipping, -1.0 if they do.
    fn direction_sign(&self) -> f64 {
        self.side_assignment
            .attacking_direction_sign(self.team_color)
    }

    /// Transform a Vector2 from world to strategy coordinates.
    pub fn world_to_strategy(&self, vec: Vector2) -> Vector2 {
        Vector2::new(vec.x * self.direction_sign(), vec.y)
    }

    /// Transform a Vector2 from strategy to world coordinates.
    pub fn strategy_to_world(&self, vec: Vector2) -> Vector2 {
        // Same operation since we're just flipping x
        Vector2::new(vec.x * self.direction_sign(), vec.y)
    }

    /// Transform an angle from world to strategy coordinates.
    pub fn angle_world_to_strategy(&self, angle: Angle) -> Angle {
        let sign = self.direction_sign();
        if sign > 0.0 {
            angle
        } else {
            // Mirror around y-axis
            if angle.radians() >= 0.0 {
                Angle::from_radians(std::f64::consts::PI - angle.radians())
            } else {
                Angle::from_radians(-std::f64::consts::PI - angle.radians())
            }
        }
    }

    /// Transform an angle from strategy to world coordinates.
    pub fn angle_strategy_to_world(&self, angle: Angle) -> Angle {
        // Same transformation since it's symmetric
        self.angle_world_to_strategy(angle)
    }

    /// Create a WorldSnapshot from TeamData for the strategy.
    ///
    /// The TeamData is already in team-relative coordinates (transformed by the tracker),
    /// but we need to convert it to the protocol types and ensure all data is normalized.
    pub fn create_world_snapshot(
        &self,
        team_data: &TeamData,
        _skill_statuses: &std::collections::HashMap<PlayerId, SkillStatus>,
        dt: f64,
    ) -> WorldSnapshot {
        // During a stoppage (Stop / ball placement), if the GC advertised the
        // upcoming restart, present that *predicted* state to the strategy so it
        // can pre-stage for it. `comply()` keys off the real state (from world
        // data), not this snapshot, so this override cannot cause a rule
        // violation. Outside the stoppage window (or with no hint) we present the
        // live state unchanged and `pre_stage` is false.
        let gs = &team_data.current_game_state;
        let in_prep_window = matches!(
            gs.game_state,
            dies_core::GameState::Stop | dies_core::GameState::BallReplacement(_)
        );
        let predicted = in_prep_window
            .then_some(gs.predicted_next_game_state)
            .flatten();
        let (game_state, us_operating, pre_stage) = match predicted {
            Some(state) => (
                state.into(),
                gs.predicted_us_operating.unwrap_or(gs.us_operating),
                true,
            ),
            None => (gs.game_state.into(), gs.us_operating, false),
        };

        WorldSnapshot {
            timestamp: team_data.t_received,
            dt,
            field_geom: team_data.field_geom.clone(),
            ball: team_data.ball.as_ref().map(|b| BallState {
                position: Vector2::new(b.position.x, b.position.y),
                velocity: Vector2::new(b.velocity.x, b.velocity.y),
                detected: b.detected,
            }),
            own_players: team_data
                .own_players
                .iter()
                .map(|p| self.convert_player_state(p, true))
                .collect(),
            opp_players: team_data
                .opp_players
                .iter()
                .map(|p| self.convert_player_state(p, false))
                .collect(),
            game_state,
            us_operating,
            pre_stage,
            our_keeper_id: team_data.current_game_state.our_keeper_id,
            freekick_kicker: team_data.current_game_state.freekick_kicker,
            double_touch_barred: team_data.current_game_state.double_touch_barred,
            possession: self.team_relative_possession(&team_data.possession.state),
            possession_stale: team_data.possession.stale,
            ball_contest: self.team_relative_contest(team_data.possession.contest.as_ref()),
        }
    }

    /// Convert the absolute (team-tagged) contest into this team's relative view,
    /// splitting the near set into our robots vs. opponents by team color.
    fn team_relative_contest(
        &self,
        contest: Option<&dies_core::BallContest>,
    ) -> Option<BallContest> {
        let contest = contest?;
        let mut ours = Vec::new();
        let mut opp = Vec::new();
        for who in &contest.near {
            if who.team_color == self.team_color {
                ours.push(who.player_id);
            } else {
                opp.push(who.player_id);
            }
        }
        Some(BallContest { ours, opp })
    }

    /// Convert the absolute (team-tagged) possession into this team's relative view.
    fn team_relative_possession(&self, state: &dies_core::PossessionState) -> Possession {
        use dies_core::PossessionState;
        match state {
            PossessionState::Loose => Possession::Loose,
            PossessionState::Contested { .. } => Possession::Contested,
            PossessionState::Owned { owner } => {
                if owner.team_color == self.team_color {
                    Possession::We(owner.player_id)
                } else {
                    Possession::Opp(owner.player_id)
                }
            }
        }
    }

    /// Convert a PlayerData to PlayerState (protocol type).
    fn convert_player_state(&self, player: &PlayerData, is_own_player: bool) -> PlayerState {
        let handicaps: HashSet<Handicap> =
            player.handicaps.iter().map(|h| h.clone().into()).collect();

        PlayerState {
            id: player.id,
            position: Vector2::new(player.position.x, player.position.y),
            velocity: Vector2::new(player.velocity.x, player.velocity.y),
            heading: player.yaw,
            angular_velocity: player.angular_speed,
            // Unified possession signal (own players only; opponents never report
            // `has_ball` to a strategy — they surface via `possession` as `Opp`).
            has_ball: is_own_player && player.has_ball,
            handicaps,
        }
    }

    /// Pass a skill command through unchanged, keeping its coordinates in the
    /// team-relative frame.
    ///
    /// Skill commands are consumed by the team controller, which runs entirely
    /// in team-relative coordinates and applies the single authoritative
    /// team→world untransform (`PlayerCmdUntransformer`) when emitting the final
    /// robot command. Flipping coordinates here as well would apply the per-color
    /// x-flip twice (`sign² = 1`), collapsing both teams onto the same side — so
    /// this is a deliberate identity on positions/headings.
    pub fn transform_skill_command(&self, cmd: &SkillCommand) -> SkillCommand {
        cmd.clone()
    }

    /// Transform a pass result from world coordinates back to strategy
    /// coordinates (only the loose-ball position carries a coordinate).
    pub fn transform_pass_result(&self, result: PassResult) -> PassResult {
        match result {
            PassResult::Failure {
                reason,
                ball_state: PassBallState::Loose { position },
            } => PassResult::Failure {
                reason,
                ball_state: PassBallState::Loose {
                    position: self.world_to_strategy(position),
                },
            },
            other => other,
        }
    }

    /// Transform debug entries from strategy coordinates to world coordinates.
    pub fn transform_debug_entries(&self, entries: Vec<DebugEntry>) -> Vec<DebugEntry> {
        entries
            .into_iter()
            .map(|entry| self.transform_debug_entry(entry))
            .collect()
    }

    /// Transform a single debug entry.
    fn transform_debug_entry(&self, entry: DebugEntry) -> DebugEntry {
        let value = match entry.value {
            DebugValue::Shape(shape) => DebugValue::Shape(self.transform_debug_shape(shape)),
            other => other,
        };
        DebugEntry {
            key: entry.key,
            value,
        }
    }

    /// Transform a debug shape from strategy to world coordinates.
    fn transform_debug_shape(&self, shape: DebugShape) -> DebugShape {
        match shape {
            DebugShape::Cross { center, color } => DebugShape::Cross {
                center: self.strategy_to_world(center),
                color,
            },
            DebugShape::Marker {
                kind,
                center,
                color,
                owner,
            } => DebugShape::Marker {
                kind,
                center: self.strategy_to_world(center),
                color,
                owner,
            },
            DebugShape::Line { start, end, color } => DebugShape::Line {
                start: self.strategy_to_world(start),
                end: self.strategy_to_world(end),
                color,
            },
            DebugShape::Circle {
                center,
                radius,
                fill,
                stroke,
            } => DebugShape::Circle {
                center: self.strategy_to_world(center),
                radius,
                fill,
                stroke,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::DebugColor;

    /// During a stoppage, a predicted restart is presented to the strategy with
    /// `pre_stage = true`; outside a stoppage the live state passes through
    /// unchanged with `pre_stage = false`.
    #[test]
    fn test_prestage_override_during_stop() {
        let tr = CoordinateTransformer::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        let empty = std::collections::HashMap::new();

        // Stop + predicted our free kick → present FreeKick, us_operating, pre_stage.
        let mut td = dies_core::mock_team_data();
        td.current_game_state.game_state = dies_core::GameState::Stop;
        td.current_game_state.us_operating = false; // symmetric-state value, ignored
        td.current_game_state.predicted_next_game_state = Some(dies_core::GameState::FreeKick);
        td.current_game_state.predicted_us_operating = Some(true);
        let snap = tr.create_world_snapshot(&td, &empty, 0.016);
        assert_eq!(snap.game_state, dies_strategy_protocol::GameState::FreeKick);
        assert!(snap.us_operating);
        assert!(snap.pre_stage);

        // No prediction → live Stop passes through, pre_stage false.
        let mut td2 = dies_core::mock_team_data();
        td2.current_game_state.game_state = dies_core::GameState::Stop;
        td2.current_game_state.predicted_next_game_state = None;
        let snap2 = tr.create_world_snapshot(&td2, &empty, 0.016);
        assert_eq!(snap2.game_state, dies_strategy_protocol::GameState::Stop);
        assert!(!snap2.pre_stage);

        // A prediction outside the stoppage window must be ignored (no override).
        let mut td3 = dies_core::mock_team_data();
        td3.current_game_state.game_state = dies_core::GameState::Run;
        td3.current_game_state.predicted_next_game_state = Some(dies_core::GameState::FreeKick);
        td3.current_game_state.predicted_us_operating = Some(true);
        let snap3 = tr.create_world_snapshot(&td3, &empty, 0.016);
        assert_eq!(snap3.game_state, dies_strategy_protocol::GameState::Run);
        assert!(!snap3.pre_stage);
    }

    #[test]
    fn test_direction_sign_blue_on_positive() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::BlueOnPositive);
        // Blue defends positive x, so attacks negative x -> sign is -1
        assert_eq!(transformer.direction_sign(), -1.0);
    }

    #[test]
    fn test_direction_sign_yellow_on_positive() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        // Yellow defends positive x, Blue attacks positive x -> sign is 1
        assert_eq!(transformer.direction_sign(), 1.0);
    }

    #[test]
    fn test_world_to_strategy_no_flip() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        let world_vec = Vector2::new(100.0, 50.0);
        let strategy_vec = transformer.world_to_strategy(world_vec);
        assert_eq!(strategy_vec, Vector2::new(100.0, 50.0));
    }

    #[test]
    fn test_world_to_strategy_with_flip() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::BlueOnPositive);
        let world_vec = Vector2::new(100.0, 50.0);
        let strategy_vec = transformer.world_to_strategy(world_vec);
        assert_eq!(strategy_vec, Vector2::new(-100.0, 50.0));
    }

    #[test]
    fn test_transform_skill_command_keeps_team_relative() {
        // Even for a team whose attacking direction is -x (sign = -1), skill
        // commands must pass through unchanged: the team controller owns the
        // single team→world untransform. Flipping here would double-apply it.
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::BlueOnPositive);
        assert_eq!(transformer.direction_sign(), -1.0);

        let cmd = SkillCommand::GoToPos {
            position: Vector2::new(1000.0, 500.0),
            heading: Some(Angle::from_radians(0.0)),
            hold_ground: false,
        };

        let transformed = transformer.transform_skill_command(&cmd);

        match transformed {
            SkillCommand::GoToPos {
                position, heading, ..
            } => {
                assert_eq!(position, Vector2::new(1000.0, 500.0));
                assert_eq!(heading, Some(Angle::from_radians(0.0)));
            }
            _ => panic!("Expected GoToPos"),
        }
    }

    #[test]
    fn test_transform_debug_shape_cross() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::BlueOnPositive);

        let shape = DebugShape::Cross {
            center: Vector2::new(100.0, 50.0),
            color: DebugColor::Red,
        };

        let transformed = transformer.transform_debug_shape(shape);

        match transformed {
            DebugShape::Cross { center, color } => {
                assert_eq!(center, Vector2::new(-100.0, 50.0));
                assert_eq!(color, DebugColor::Red);
            }
            _ => panic!("Expected Cross"),
        }
    }
}
