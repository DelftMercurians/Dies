use crate::TeamColor;
use serde::{Deserialize, Serialize};
use std::hash::Hash;
use typeshare::typeshare;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[typeshare(serialized_as = "u32")]
pub struct TeamId(pub u32);

impl TeamId {
    /// Create a TeamId from a name. This is deterministic and unique for a given name (case insensitive and whitespace ignored).
    pub fn from_name(name: &str) -> Self {
        // FNV-1a algorithm
        let mut hash: u32 = 2166136261; // FNV offset basis
        for byte in name
            .to_lowercase()
            .chars()
            .filter(|c| !c.is_whitespace())
            .flat_map(|c| c.to_string().into_bytes())
        {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(16777619); // FNV prime
        }
        TeamId(hash)
    }
}

impl From<&str> for TeamId {
    fn from(name: &str) -> Self {
        TeamId::from_name(name)
    }
}

impl From<String> for TeamId {
    fn from(name: String) -> Self {
        TeamId::from_name(&name)
    }
}

impl std::fmt::Display for TeamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct TeamInfo {
    pub id: TeamId,
    pub name: Option<String>,
}

impl TeamInfo {
    pub fn new(id: TeamId) -> Self {
        Self { id, name: None }
    }

    pub fn new_with_name(name: &str) -> Self {
        Self {
            id: TeamId::from_name(name),
            name: Some(name.to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[typeshare]
pub struct TeamConfiguration {
    team_a_color: TeamColor,
    team_a_info: TeamInfo,
    team_b_color: TeamColor,
    team_b_info: TeamInfo,
}

impl TeamConfiguration {
    pub fn new(blue_team: TeamInfo, yellow_team: TeamInfo) -> Self {
        Self {
            team_a_color: TeamColor::Blue,
            team_a_info: blue_team,
            team_b_color: TeamColor::Yellow,
            team_b_info: yellow_team,
        }
    }

    pub fn get_team_id(&self, team_color: TeamColor) -> TeamId {
        if team_color == self.team_a_color {
            self.team_a_info.id
        } else {
            self.team_b_info.id
        }
    }

    /// Get the team info for a given team id.
    ///
    /// # Panics
    ///
    /// Panics if the team id is not found in the configuration.
    pub fn get_team_info(&self, team_id: TeamId) -> &TeamInfo {
        if self.team_a_info.id == team_id {
            &self.team_a_info
        } else if self.team_b_info.id == team_id {
            &self.team_b_info
        } else {
            panic!("Team {} not found in configuration", team_id);
        }
    }

    /// Get the team color for a given team id.
    ///
    /// # Panics
    ///
    /// Panics if the team id is not found in the configuration.
    pub fn get_team_color(&self, team_id: TeamId) -> TeamColor {
        if self.team_a_info.id == team_id {
            self.team_a_color
        } else if self.team_b_info.id == team_id {
            self.team_b_color
        } else {
            panic!("Team {} not found in configuration", team_id);
        }
    }

    /// Switch the colors of the two teams.
    pub fn switch_colors(&mut self) {
        std::mem::swap(&mut self.team_a_color, &mut self.team_b_color);
    }
}
