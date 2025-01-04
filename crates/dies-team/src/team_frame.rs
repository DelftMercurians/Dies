use dies_core::{BallFrame, GameStateType, PlayerFrame, TeamColor, WorldFrame};

use crate::utils::SideTransform;

/// World state from a single frame, transformed to a specific team's perspective.
///
/// In the transformed frame, the team's opponent's goal is always towards the positive
/// x-axis -- we are always attacking towards the positive x-axis.
#[derive(Debug, Clone)]
pub struct TeamFrame {
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    pub own_players: Vec<PlayerFrame>,
    pub opp_players: Vec<PlayerFrame>,
    pub ball: Option<BallFrame>,
    pub game_state: TeamGameState,
}

impl TeamFrame {
    pub fn from_world_frame(world_frame: &WorldFrame, team: TeamColor) -> Self {
        let side = SideTransform::new(team, world_frame.side_assignment);
        let own_team = world_frame
            .get_team(team)
            .iter()
            .map(|p| side.transform_player(p))
            .collect();
        let opp_side = SideTransform::new(team.opponent(), world_frame.side_assignment);
        let opp_team = world_frame
            .get_team(team.opponent())
            .iter()
            .map(|p| opp_side.transform_player(p))
            .collect();
        let ball = world_frame.ball.as_ref().map(|b| side.transform_ball(b));
        Self {
            dt: world_frame.dt,
            own_players: own_team,
            opp_players: opp_team,
            ball,
            game_state: side.transform_game_state(&world_frame.game_state),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TeamGameState {
    pub game_state: GameStateType,
    pub us_operating: bool,
}

#[cfg(test)]
mod tests {
    use dies_core::{
        mock_world_frame, Angle, GameStateInfo, GameStateType, SideAssignment, Vector2, Vector3,
    };

    use super::*;
    use std::f64::consts::PI;

    fn create_test_world_frame(side_assignment: SideAssignment) -> WorldFrame {
        let mut frame = mock_world_frame();
        frame.side_assignment = side_assignment;
        frame
    }

    #[test]
    fn test_blue_team_standard_orientation() {
        // Test when blue attacks positive x (standard orientation)
        let world_frame = create_test_world_frame(SideAssignment::BluePositive);
        let team_frame = TeamFrame::from_world_frame(&world_frame, TeamColor::Blue);

        // Own team (blue) positions and orientations should be unchanged
        let own_player = &team_frame.own_players[0];
        assert_eq!(own_player.position, Vector2::new(2000.0, 1000.0));
        assert_eq!(own_player.velocity, Vector2::new(1000.0, 500.0));
        assert_eq!(own_player.yaw.radians(), PI / 4.0);
        assert_eq!(own_player.angular_speed, 2.0);

        // Opponent team (yellow) positions and orientations should be unchanged
        let opp_player = &team_frame.opp_players[0];
        assert_eq!(opp_player.position, Vector2::new(-2000.0, -1000.0));
        assert_eq!(opp_player.velocity, Vector2::new(-1000.0, -500.0));
        assert_eq!(opp_player.yaw.radians(), -PI / 4.0);
        assert_eq!(opp_player.angular_speed, -2.0);

        // Ball should be unchanged
        let ball = team_frame.ball.unwrap();
        assert_eq!(ball.position, Vector3::new(1000.0, 500.0, 0.0));
        assert_eq!(ball.velocity, Vector3::new(2000.0, -1000.0, 0.0));

        // Game state
        assert!(matches!(
            team_frame.game_state.game_state,
            GameStateType::BallPlacement(pos) if pos == Vector2::new(1500.0, 750.0)
        ));
        assert!(team_frame.game_state.us_operating);
    }

    #[test]
    fn test_yellow_team_flipped_orientation() {
        // Test when blue attacks positive x (yellow needs transformation)
        let world_frame = create_test_world_frame(SideAssignment::BluePositive);
        let team_frame = TeamFrame::from_world_frame(&world_frame, TeamColor::Yellow);

        // Own team (yellow) positions and orientations should be flipped
        let own_player = &team_frame.own_players[0];
        assert_eq!(own_player.position, Vector2::new(2000.0, 1000.0)); // Flipped from -2000.0
        assert_eq!(own_player.velocity, Vector2::new(1000.0, 500.0)); // Flipped from -1000.0
        assert_eq!(own_player.yaw.radians(), PI - (-PI / 4.0)); // Flipped around y-axis
        assert_eq!(own_player.angular_speed, 2.0); // Sign flipped from -2.0

        // Opponent team (blue) positions and orientations should be flipped
        let opp_player = &team_frame.opp_players[0];
        assert_eq!(opp_player.position, Vector2::new(-2000.0, -1000.0)); // Flipped from 2000.0
        assert_eq!(opp_player.velocity, Vector2::new(-1000.0, -500.0)); // Flipped from 1000.0
        assert_eq!(opp_player.yaw.radians(), PI - (PI / 4.0)); // Flipped around y-axis
        assert_eq!(opp_player.angular_speed, -2.0); // Sign flipped

        // Ball should be flipped
        let ball = team_frame.ball.unwrap();
        assert_eq!(ball.position, Vector3::new(-1000.0, 500.0, 0.0));
        assert_eq!(ball.velocity, Vector3::new(-2000.0, -1000.0, 0.0));

        // Game state
        assert!(matches!(
            team_frame.game_state.game_state,
            GameStateType::BallPlacement(pos) if pos == Vector2::new(-1500.0, 750.0)
        ));
        assert!(!team_frame.game_state.us_operating);
    }

    #[test]
    fn test_angle_transformations() {
        let side = SideTransform::new(TeamColor::Yellow, SideAssignment::BluePositive);

        // Test all quadrants
        let angles = vec![
            (0.0, PI),                    // 0° -> 180°
            (PI / 2.0, PI / 2.0),         // 90° -> 90° (unchanged)
            (PI, 0.0),                    // 180° -> 0°
            (-PI / 2.0, -PI / 2.0),       // -90° -> -90° (unchanged)
            (PI / 4.0, 3.0 * PI / 4.0),   // 45° -> 135°
            (-PI / 4.0, -3.0 * PI / 4.0), // -45° -> -135°
        ];

        for (input, expected) in angles {
            let result = side.transform_angle(Angle::from_radians(input));
            assert!(
                (result.radians() - expected).abs() < 1e-10,
                "Failed for input angle {} rad: expected {} rad, got {} rad",
                input,
                expected,
                result.radians()
            );
        }
    }

    #[test]
    fn test_game_state_transformations() {
        let side = SideTransform::new(TeamColor::Yellow, SideAssignment::BluePositive);

        // Test symmetric states (shouldn't change except for operating team)
        let symmetric_states = vec![
            GameStateType::Halt,
            GameStateType::Stop,
            GameStateType::Run,
            GameStateType::Timeout,
        ];

        for state_type in symmetric_states {
            let state = GameStateInfo {
                state_type,
                operating_team: None,
            };
            let result = side.transform_game_state(&state);
            assert_eq!(result.game_state, state_type);
            assert!(!result.us_operating);
        }

        // Test ball placement transformation
        let placement_pos = Vector2::new(1000.0, 500.0);
        let state = GameStateInfo {
            state_type: GameStateType::BallPlacement(placement_pos),
            operating_team: Some(TeamColor::Blue),
        };
        let result = side.transform_game_state(&state);
        if let GameStateType::BallPlacement(transformed_pos) = result.game_state {
            assert_eq!(transformed_pos, Vector2::new(-1000.0, 500.0));
        } else {
            panic!("Expected BallPlacement state");
        }
        assert!(!result.us_operating);
    }
}
