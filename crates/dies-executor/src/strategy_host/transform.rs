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
    BallState, DebugEntry, DebugShape, DebugValue, Handicap, PlayerState,
    PlayerId, SkillCommand, SkillStatus, WorldSnapshot,
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
            game_state: team_data.current_game_state.game_state.into(),
            us_operating: team_data.current_game_state.us_operating,
            our_keeper_id: team_data.current_game_state.our_keeper_id,
        }
    }

    /// Convert a PlayerData to PlayerState (protocol type).
    fn convert_player_state(&self, player: &PlayerData, is_own_player: bool) -> PlayerState {
        let handicaps: HashSet<Handicap> = player
            .handicaps
            .iter()
            .map(|h| h.clone().into())
            .collect();

        PlayerState {
            id: player.id,
            position: Vector2::new(player.position.x, player.position.y),
            velocity: Vector2::new(player.velocity.x, player.velocity.y),
            heading: player.yaw,
            angular_velocity: player.angular_speed,
            has_ball: if is_own_player {
                player.breakbeam_ball_detected
            } else {
                false
            },
            handicaps,
        }
    }

    /// Transform a skill command from strategy coordinates to world coordinates.
    pub fn transform_skill_command(&self, cmd: &SkillCommand) -> SkillCommand {
        match cmd {
            SkillCommand::GoToPos { position, heading } => SkillCommand::GoToPos {
                position: self.strategy_to_world(*position),
                heading: heading.map(|h| self.angle_strategy_to_world(h)),
            },
            SkillCommand::Dribble {
                target_pos,
                target_heading,
            } => SkillCommand::Dribble {
                target_pos: self.strategy_to_world(*target_pos),
                target_heading: self.angle_strategy_to_world(*target_heading),
            },
            SkillCommand::PickupBall { target_heading } => SkillCommand::PickupBall {
                target_heading: self.angle_strategy_to_world(*target_heading),
            },
            SkillCommand::ReflexShoot { target } => SkillCommand::ReflexShoot {
                target: self.strategy_to_world(*target),
            },
            SkillCommand::Stop => SkillCommand::Stop,
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
    fn test_transform_skill_command_go_to_pos() {
        let transformer =
            CoordinateTransformer::new(TeamColor::Blue, SideAssignment::BlueOnPositive);
        
        let cmd = SkillCommand::GoToPos {
            position: Vector2::new(1000.0, 500.0),
            heading: Some(Angle::from_radians(0.0)),
        };
        
        let transformed = transformer.transform_skill_command(&cmd);
        
        match transformed {
            SkillCommand::GoToPos { position, heading } => {
                assert_eq!(position, Vector2::new(-1000.0, 500.0));
                // 0 radians flipped -> PI radians
                assert!(heading.is_some());
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

