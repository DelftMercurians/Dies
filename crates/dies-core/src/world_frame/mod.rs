mod ball_frame;
mod game_state;
mod player_frame;

pub use ball_frame::*;
pub use game_state::*;
pub use player_frame::*;

use crate::FieldGeometry;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct DiesInstant(f64);

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
    pub blue_team: Vec<PlayerFrame>,
    pub yellow_team: Vec<PlayerFrame>,
    pub ball: Option<BallFrame>,
    pub field_geom: Option<FieldGeometry>,
    pub current_game_state: GameState,
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
