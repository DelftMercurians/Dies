use serde::{Deserialize, Serialize};

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
