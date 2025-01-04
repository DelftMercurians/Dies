mod ball_frame;
mod game_state;
mod player_frame;

use std::time::Duration;

pub use ball_frame::*;
pub use game_state::*;
pub use player_frame::*;

use crate::{Angle, PlayerId, Vector2, Vector3};
use serde::{Deserialize, Serialize};

/// A point in time represented as a number of seconds since the start of the application.
///
/// This instant is guaranteed to be:
///  - monotonically increasing
///  - non-negative
///  - finite and non-NaN
#[derive(Serialize, Deserialize, Clone, Debug, Copy, Default)]
pub struct DiesInstant(f64);

impl DiesInstant {
    fn new(value: f64) -> Self {
        assert!(value >= 0.0, "DiesInstant must be non-negative");
        assert!(value.is_finite(), "DiesInstant must be finite");
        Self(value)
    }

    /// Get the underlying floating point value.
    pub fn as_secs_f64(&self) -> f64 {
        self.0
    }

    /// Get the duration between this instant and another instant.
    ///
    /// If the other instant is after this instant, the result is 0, therefore this
    /// value is guaranteed to be non-negative.
    pub fn duration_since(&self, other: &Self) -> f64 {
        if self.0 < other.0 {
            return 0.0;
        }
        self.0 - other.0
    }

    pub fn test_value(value: f64) -> Self {
        Self::new(value)
    }
}

impl std::ops::Add<f64> for DiesInstant {
    type Output = Self;

    fn add(self, rhs: f64) -> Self::Output {
        assert!(rhs >= 0.0, "DiesInstant cannot decrease");
        Self::new(self.0 + rhs)
    }
}

impl std::ops::Add<Duration> for DiesInstant {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        Self::new(self.0 + rhs.as_secs_f64())
    }
}

impl std::ops::AddAssign<f64> for DiesInstant {
    fn add_assign(&mut self, rhs: f64) {
        assert!(rhs >= 0.0, "DiesInstant cannot decrease");
        *self = Self::new(self.0 + rhs);
    }
}

impl std::ops::AddAssign<Duration> for DiesInstant {
    fn add_assign(&mut self, rhs: Duration) {
        *self = Self::new(self.0 + rhs.as_secs_f64());
    }
}

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

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TeamColor {
    Blue,
    Yellow,
}

impl TeamColor {
    pub fn opponent(&self) -> TeamColor {
        match self {
            TeamColor::Blue => TeamColor::Yellow,
            TeamColor::Yellow => TeamColor::Blue,
        }
    }
}

/// The side the two teams are assigned to.
///
/// `BluePositive` means that the blue team's goal is the positive side of the field.
/// `YellowPositive` means that the yellow team's goal is the positive side of the field.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideAssignment {
    BluePositive,
    YellowPositive,
}

impl SideAssignment {
    /// Get the sign of the x-coordinate for the blue team's goal's side.
    pub fn blue_goal_side(&self) -> f64 {
        match self {
            SideAssignment::BluePositive => 1.0,
            SideAssignment::YellowPositive => -1.0,
        }
    }

    /// Get the sign of the x-coordinate for the yellow team's goal's side.
    pub fn yellow_goal_side(&self) -> f64 {
        match self {
            SideAssignment::BluePositive => -1.0,
            SideAssignment::YellowPositive => 1.0,
        }
    }
}

/// Role type that determines special rules applied to a player.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RoleType {
    /// No special role.
    #[default]
    None,
    /// A player who is the goalkeeper.
    Goalkeeper,
    /// A player who is the kickoff kicker.
    KickoffKicker,
    /// A player who is the freekick taker.
    FreekickTaker,
    /// A player who is the penalty taker.
    PenaltyTaker,
}

pub fn mock_world_frame() -> WorldFrame {
    WorldFrame {
        timestamp: DiesInstant(0.0),
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
