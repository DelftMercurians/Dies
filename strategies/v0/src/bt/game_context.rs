//! `GameContext` — what a strategy is handed each frame to declare its roles.
//!
//! Mirrors the old executor-side context, but reads its game-state / ball-timing
//! facts from values the runtime threads in from the IPC `World` rather than from
//! a shared `TeamData`.

use dies_strategy_protocol::{GameState, PlayerId};

use super::role_assignment::{RoleAssignmentProblem, RoleBuilder};

pub struct GameContext {
    game_state: GameState,
    us_operating: bool,
    our_keeper_id: Option<PlayerId>,
    /// Seconds the ball has continuously been on our half, if it is.
    ball_on_our_side: Option<f64>,
    /// Seconds the ball has continuously been on the opponent half, if it is.
    ball_on_opp_side: Option<f64>,
    role_builders: Vec<RoleBuilder>,
}

impl GameContext {
    pub fn new(
        game_state: GameState,
        us_operating: bool,
        our_keeper_id: Option<PlayerId>,
        ball_on_our_side: Option<f64>,
        ball_on_opp_side: Option<f64>,
    ) -> Self {
        Self {
            game_state,
            us_operating,
            our_keeper_id,
            ball_on_our_side,
            ball_on_opp_side,
            role_builders: Vec::new(),
        }
    }

    pub fn game_state(&self) -> GameState {
        self.game_state
    }

    pub fn us_operating(&self) -> bool {
        self.us_operating
    }

    pub fn our_keeper_id(&self) -> Option<PlayerId> {
        self.our_keeper_id
    }

    pub fn ball_has_been_on_our_side_for_at_least(&self, secs: f64) -> bool {
        self.ball_on_our_side.map(|t| t >= secs).unwrap_or(false)
    }

    pub fn ball_has_been_on_opp_side_for_at_least(&self, secs: f64) -> bool {
        self.ball_on_opp_side.map(|t| t >= secs).unwrap_or(false)
    }

    /// Declare a role and return a builder to configure it.
    pub fn add_role(&mut self, name: &str) -> &mut RoleBuilder {
        let index = self.role_builders.len();
        self.role_builders.push(RoleBuilder::new(name, index));
        &mut self.role_builders[index]
    }

    /// Collapse the declared roles into the assignment problem.
    pub fn into_role_assignment_problem(self) -> RoleAssignmentProblem {
        let roles = self
            .role_builders
            .into_iter()
            .filter_map(|b| b.build().ok())
            .collect();
        RoleAssignmentProblem { roles }
    }
}
