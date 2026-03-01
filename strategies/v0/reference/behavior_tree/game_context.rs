use std::{sync::Arc, time::Duration};

use dies_core::{GameStateData, TeamData};

use crate::behavior_tree::{RoleAssignmentProblem, RoleBuilder};

pub struct GameContext {
    game_state: GameStateData,
    role_builders: Vec<RoleBuilder>,
    team_data: Arc<TeamData>,
}

impl GameContext {
    pub fn new(team_data: Arc<TeamData>) -> Self {
        Self {
            game_state: team_data.current_game_state.clone(),
            role_builders: Vec::new(),
            team_data,
        }
    }

    pub fn team_data(&self) -> Arc<TeamData> {
        self.team_data.clone()
    }

    pub fn ball_has_been_on_our_side_for_at_least(&self, secs: f64) -> bool {
        self.team_data
            .ball_on_our_side
            .map(|t| t > Duration::from_secs_f64(secs))
            .unwrap_or(false)
    }

    pub fn ball_has_been_on_opp_side_for_at_least(&self, secs: f64) -> bool {
        self.team_data
            .ball_on_opp_side
            .map(|t| t > Duration::from_secs_f64(secs))
            .unwrap_or(false)
    }

    pub fn game_state(&self) -> GameStateData {
        self.game_state.clone()
    }

    /// Add a role and return a builder for configuration
    pub fn add_role(&mut self, name: &str) -> &mut RoleBuilder {
        let last_index = self.role_builders.len();
        let builder = RoleBuilder::new(name, last_index);
        self.role_builders.push(builder);
        &mut self.role_builders[last_index]
    }

    /// Extract the final role assignment problem
    pub fn into_role_assignment_problem(self) -> RoleAssignmentProblem {
        let roles = self
            .role_builders
            .into_iter()
            .filter_map(|builder| builder.build().ok())
            .collect();

        RoleAssignmentProblem { roles }
    }
}
