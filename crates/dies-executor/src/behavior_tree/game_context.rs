use dies_core::{FieldGeometry, GameStateData, TeamData};

use crate::behavior_tree::{RoleAssignmentProblem, RoleBuilder};

pub struct GameContext {
    game_state: GameStateData,
    num_own_players: usize,
    num_opp_players: usize,
    field_geom: Option<FieldGeometry>,
    role_builders: Vec<RoleBuilder>,
}

impl GameContext {
    pub fn new(team_data: &TeamData) -> Self {
        Self {
            game_state: team_data.current_game_state.clone(),
            num_own_players: team_data.own_players.len(),
            num_opp_players: team_data.opp_players.len(),
            field_geom: team_data.field_geom.clone(),
            role_builders: Vec::new(),
        }
    }

    pub fn game_state(&self) -> GameStateData {
        self.game_state.clone()
    }

    /// Add a role and return a builder for configuration
    pub fn add_role(&mut self, name: &str) -> &mut RoleBuilder {
        let builder = RoleBuilder::new(name);
        self.role_builders.push(builder);
        let last_index = self.role_builders.len() - 1;
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
