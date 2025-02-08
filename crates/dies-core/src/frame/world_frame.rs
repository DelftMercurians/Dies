use crate::{Angle, PlayerId, Vector2, Vector3};
use serde::{Deserialize, Serialize};

use super::{
    ball_frame::BallFrame,
    dies_instant::DiesInstant,
    game_state::{GameStateInfo, GameStateType},
    player_frame::{PlayerFeedback, PlayerFrame},
    team_color::{SideAssignment, TeamColor},
};

/// A struct to store the world state from a single frame.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WorldFrame {
    /// The number of seconds between the start of the run and the processing of this
    /// frame.
    ///
    /// This timestamp should be the basis for all timekeeping throughout the framework.
    pub timestamp: DiesInstant,
    /// The time since the last frame was received, in seconds
    pub dt: f64,
    /// Blue players
    pub blue_team: Vec<PlayerFrame>,
    /// Yellow players
    pub yellow_team: Vec<PlayerFrame>,
    /// The ball, if it has been detected at least once
    pub ball: Option<BallFrame>,
    /// The current game state - Freekick, Run, Timeout, etc.
    pub game_state: GameStateInfo,
    /// The side assignment for the field - `BluePositive` or `YellowPositive`
    pub side_assignment: SideAssignment,
}

impl WorldFrame {
    pub fn get_team(&self, team: TeamColor) -> &Vec<PlayerFrame> {
        match team {
            TeamColor::Blue => &self.blue_team,
            TeamColor::Yellow => &self.yellow_team,
        }
    }
}

pub fn mock_world_frame() -> WorldFrame {
    WorldFrame {
        timestamp: DiesInstant::test_value(0.0),
        dt: 0.0,
        blue_team: vec![PlayerFrame {
            id: PlayerId::new(0),
            position: Vector2::new(1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }],
        yellow_team: vec![PlayerFrame {
            id: PlayerId::new(0),
            position: Vector2::new(-1000.0, 1000.0),
            velocity: Vector2::zeros(),
            yaw: Angle::default(),
            angular_speed: 0.0,
            feedback: PlayerFeedback::NotControlled,
        }],
        ball: Some(BallFrame {
            position: Vector3::zeros(),
            velocity: Vector3::zeros(),
            detected: true,
        }),
        game_state: GameStateInfo {
            state_type: GameStateType::Run,
            operating_team: None,
        },
        side_assignment: SideAssignment::BluePositive,
    }
}
