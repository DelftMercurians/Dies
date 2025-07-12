use anyhow::{anyhow, Result};
use dies_core::{PlayerId, TeamData};
use lapjv::lapjv;
use ndarray::Array2;
use rhai::Engine;
use std::collections::HashMap;
use std::sync::Arc;

use crate::behavior_tree::bt_callback::BtCallback;
use crate::behavior_tree::RhaiBehaviorNode;

use super::RobotSituation;

/// Represents a role that can be assigned to robots
#[derive(Clone)]
pub struct Role {
    pub name: String,
    pub min_count: usize,
    pub max_count: usize,
    pub scorer: BtCallback<f64>,
    pub require_filter: Option<BtCallback<bool>>,
    pub exclude_filter: Option<BtCallback<bool>>,
    pub tree_builder: BtCallback<RhaiBehaviorNode>,
}

impl std::fmt::Debug for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Role {{ name: {}, min_count: {}, max_count: {}, scorer: {:?}, require_filter: {:?}, exclude_filter: {:?} }}", self.name, self.min_count, self.max_count, self.scorer, self.require_filter, self.exclude_filter)
    }
}

/// The complete role assignment problem definition
#[derive(Clone, Debug)]
pub struct RoleAssignmentProblem {
    pub roles: Vec<Role>,
}

/// Builder for creating roles with a fluent API
#[derive(Clone)]
pub struct RoleBuilder {
    name: String,
    min_count: usize,
    max_count: usize,
    scorer: Option<BtCallback<f64>>,
    require_filter: Option<BtCallback<bool>>,
    exclude_filter: Option<BtCallback<bool>>,
    tree_builder: Option<BtCallback<RhaiBehaviorNode>>,
}

impl RoleBuilder {
    /// Create a new role builder
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            min_count: 0,
            max_count: usize::MAX,
            scorer: None,
            require_filter: None,
            exclude_filter: None,
            tree_builder: None,
        }
    }

    /// Set minimum number of robots for this role
    pub fn min(mut self, count: usize) -> Self {
        self.min_count = count;
        self
    }

    /// Set maximum number of robots for this role
    pub fn max(mut self, count: usize) -> Self {
        self.max_count = count;
        self
    }

    /// Set exact count (min = max)
    pub fn count(mut self, count: usize) -> Self {
        self.min_count = count;
        self.max_count = count;
        self
    }

    /// Set the scoring function for this role
    pub fn score(mut self, scorer: BtCallback<f64>) -> Self {
        self.scorer = Some(scorer);
        self
    }

    /// Set a filter function that must return true for a robot to be considered for this role
    pub fn require(mut self, filter: BtCallback<bool>) -> Self {
        self.require_filter = Some(filter);
        self
    }

    /// Set a filter function that must return false for a robot to be considered for this role
    pub fn exclude(mut self, filter: BtCallback<bool>) -> Self {
        self.exclude_filter = Some(filter);
        self
    }

    /// Set the behavior tree builder
    pub fn behavior(mut self, builder: BtCallback<RhaiBehaviorNode>) -> Self {
        self.tree_builder = Some(builder);
        self
    }

    /// Build the role
    pub fn build(self) -> Result<Role> {
        let scorer = self
            .scorer
            .ok_or_else(|| anyhow!("Role '{}' missing scorer function", self.name))?;

        let tree_builder = self
            .tree_builder
            .ok_or_else(|| anyhow!("Role '{}' missing behavior tree builder", self.name))?;

        Ok(Role {
            name: self.name,
            min_count: self.min_count,
            max_count: self.max_count,
            scorer,
            require_filter: self.require_filter,
            exclude_filter: self.exclude_filter,
            tree_builder,
        })
    }
}

/// Solver for the role assignment problem using the lapjv algorithm
pub struct RoleAssignmentSolver {
    // Cache for performance
    score_matrix_buffer: Vec<f64>,
}

impl RoleAssignmentSolver {
    pub fn new() -> Self {
        Self {
            score_matrix_buffer: Vec::with_capacity(100),
        }
    }

    /// Solve the role assignment problem
    pub fn solve(
        &mut self,
        problem: &RoleAssignmentProblem,
        active_robots: &[PlayerId],
        team_data: &TeamData,
        engine: &Engine,
    ) -> Result<HashMap<PlayerId, String>> {
        if active_robots.is_empty() {
            return Ok(HashMap::new());
        }

        // Create expanded assignment matrix
        let (cost_matrix, role_mapping) =
            self.build_cost_matrix(problem, active_robots, team_data, engine)?;

        // Solve using lapjv (minimizes cost, so we negate scores)
        let (_row_sol, col_sol) =
            lapjv(&cost_matrix).map_err(|e| anyhow!("Failed to solve assignment: {:?}", e))?;

        // Convert solution to robot->role mapping
        let mut assignments = HashMap::new();
        for (robot_idx, &role_idx) in col_sol.iter().enumerate() {
            if robot_idx < active_robots.len() && role_idx < role_mapping.len() {
                let robot_id = active_robots[robot_idx];
                let role_name = &role_mapping[role_idx];
                assignments.insert(robot_id, role_name.clone());
            }
        }

        // Verify constraints are satisfied
        self.verify_assignments(&assignments, problem)?;

        Ok(assignments)
    }

    /// Build cost matrix for the assignment problem
    fn build_cost_matrix(
        &mut self,
        problem: &RoleAssignmentProblem,
        robots: &[PlayerId],
        team_data: &TeamData,
        engine: &Engine,
    ) -> Result<(Array2<f64>, Vec<String>)> {
        let n_robots = robots.len();

        // Calculate total slots needed
        let total_slots: usize = problem
            .roles
            .iter()
            .map(|r| r.max_count.min(n_robots))
            .sum();

        // Build role slot mapping
        let mut role_mapping = Vec::with_capacity(total_slots);
        for role in &problem.roles {
            for _ in 0..role.max_count.min(n_robots) {
                role_mapping.push(role.name.clone());
            }
        }

        // Make matrix square by padding if necessary
        let matrix_size = n_robots.max(role_mapping.len());
        let mut cost_matrix = Array2::from_elem((matrix_size, matrix_size), 1e9);

        // Fill in actual costs
        for (i, &robot_id) in robots.iter().enumerate() {
            let situation = self.create_robot_situation(robot_id, team_data);

            let mut role_start_idx = 0;
            for role in &problem.roles {
                let slots = role.max_count.min(n_robots);

                // Check filters
                if self.violates_filters(&role, &situation, engine) {
                    // High cost for filter violations
                    for j in 0..slots {
                        cost_matrix[[i, role_start_idx + j]] = 1e8;
                    }
                } else {
                    // Calculate score and convert to cost (negate for minimization)
                    let score = role.scorer.call(&situation, engine)?;
                    let cost = -score;

                    for j in 0..slots {
                        cost_matrix[[i, role_start_idx + j]] = cost;
                    }
                }

                role_start_idx += slots;
            }
        }

        Ok((cost_matrix, role_mapping))
    }

    /// Check if a robot violates role filters
    fn violates_filters(&self, role: &Role, situation: &RobotSituation, engine: &Engine) -> bool {
        // Check require filter - must return true
        if let Some(ref require_filter) = role.require_filter {
            if !require_filter.call(situation, engine).unwrap_or(false) {
                return true;
            }
        }

        // Check exclude filter - must return false
        if let Some(ref exclude_filter) = role.exclude_filter {
            if exclude_filter.call(situation, engine).unwrap_or(false) {
                return true;
            }
        }

        false
    }

    /// Create robot situation for scoring
    fn create_robot_situation(&self, robot_id: PlayerId, team_data: &TeamData) -> RobotSituation {
        RobotSituation::new(
            robot_id,
            std::sync::Arc::new(team_data.clone()),
            Default::default(),
            String::new(),
        )
    }

    /// Verify that assignments satisfy all constraints
    fn verify_assignments(
        &self,
        assignments: &HashMap<PlayerId, String>,
        problem: &RoleAssignmentProblem,
    ) -> Result<()> {
        // Count assignments per role
        let mut role_counts: HashMap<&str, usize> = HashMap::new();
        for role_name in assignments.values() {
            *role_counts.entry(role_name.as_str()).or_insert(0) += 1;
        }

        // Check min/max constraints
        for role in &problem.roles {
            let count = role_counts.get(role.name.as_str()).copied().unwrap_or(0);

            if count < role.min_count {
                return Err(anyhow!(
                    "Role '{}' has {} robots but requires at least {}",
                    role.name,
                    count,
                    role.min_count
                ));
            }

            if count > role.max_count {
                return Err(anyhow!(
                    "Role '{}' has {} robots but allows at most {}",
                    role.name,
                    count,
                    role.max_count
                ));
            }
        }

        Ok(())
    }
}

impl Default for RoleAssignmentSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_core::{
        Angle, BallData, GameState, GameStateData, PlayerData, SysStatus, TeamData, Vector2,
        Vector3,
    };
    use std::sync::Arc;

    fn create_test_team_data(num_players: usize) -> TeamData {
        let mut own_players = Vec::new();
        for i in 0..num_players {
            own_players.push(PlayerData {
                timestamp: 0.0,
                id: PlayerId::new(i as u32),
                raw_position: Vector2::new(i as f64 * 100.0, 0.0),
                position: Vector2::new(i as f64 * 100.0, 0.0),
                velocity: Vector2::zeros(),
                yaw: Angle::from_radians(0.0),
                raw_yaw: Angle::from_radians(0.0),
                angular_speed: 0.0,
                primary_status: Some(SysStatus::Ready),
                kicker_cap_voltage: Some(0.0),
                kicker_temp: Some(0.0),
                pack_voltages: Some([0.0, 0.0]),
                breakbeam_ball_detected: false,
                imu_status: Some(SysStatus::Ready),
                kicker_status: Some(SysStatus::Standby),
            });
        }

        TeamData {
            t_received: 0.0,
            t_capture: 0.0,
            dt: 0.01,
            own_players,
            opp_players: vec![],
            ball: Some(BallData {
                timestamp: 0.0,
                position: Vector3::new(0.0, 0.0, 0.0),
                raw_position: vec![Vector3::new(0.0, 0.0, 0.0)],
                velocity: Vector3::zeros(),
                detected: true,
            }),
            current_game_state: GameStateData {
                game_state: GameState::Stop,
                us_operating: false,
            },
            field_geom: None,
        }
    }

    fn create_test_role(name: &str, min: usize, max: usize) -> Role {
        RoleBuilder::new(name)
            .min(min)
            .max(max)
            .score(BtCallback::new_native(|_| 50.0))
            .behavior(BtCallback::new_native(|_| {
                super::super::rhai_types::RhaiBehaviorNode(
                    super::super::bt_node::BehaviorNode::Noop(
                        super::super::bt_node::NoopNode::new(),
                    ),
                )
            }))
            .build()
            .unwrap()
    }

    #[test]
    fn test_role_builder() {
        let role = RoleBuilder::new("attacker")
            .min(1)
            .max(2)
            .score(BtCallback::new_native(|_| 100.0))
            .require(BtCallback::new_native(|s| s.player_id == PlayerId::new(1)))
            .exclude(BtCallback::new_native(|s| s.player_id == PlayerId::new(0)))
            .behavior(BtCallback::new_native(|_| {
                super::super::rhai_types::RhaiBehaviorNode(
                    super::super::bt_node::BehaviorNode::Noop(
                        super::super::bt_node::NoopNode::new(),
                    ),
                )
            }))
            .build()
            .unwrap();

        assert_eq!(role.name, "attacker");
        assert_eq!(role.min_count, 1);
        assert_eq!(role.max_count, 2);
        assert!(role.require_filter.is_some());
        assert!(role.exclude_filter.is_some());
    }

    #[test]
    fn test_simple_assignment() {
        let mut solver = RoleAssignmentSolver::new();
        let team_data = create_test_team_data(3);
        let active_robots = vec![PlayerId::new(0), PlayerId::new(1), PlayerId::new(2)];
        let engine = rhai::Engine::new();

        let problem = RoleAssignmentProblem {
            roles: vec![
                create_test_role("goalkeeper", 1, 1),
                create_test_role("attacker", 1, 1),
                create_test_role("defender", 1, 1),
            ],
        };

        let assignments = solver
            .solve(&problem, &active_robots, &team_data, &engine)
            .unwrap();

        assert_eq!(assignments.len(), 3);
        // All robots should be assigned
    }

    #[test]
    fn test_filter_constraints() {
        let mut solver = RoleAssignmentSolver::new();
        let team_data = create_test_team_data(4);
        let active_robots = vec![
            PlayerId::new(0),
            PlayerId::new(1),
            PlayerId::new(2),
            PlayerId::new(3),
        ];
        let engine = rhai::Engine::new();

        let goalkeeper_role = RoleBuilder::new("goalkeeper")
            .count(1)
            .score(BtCallback::new_native(|_| 100.0))
            .require(BtCallback::new_native(|s| s.player_id == PlayerId::new(0)))
            .behavior(BtCallback::new_native(|_| {
                super::super::rhai_types::RhaiBehaviorNode(
                    super::super::bt_node::BehaviorNode::Noop(
                        super::super::bt_node::NoopNode::new(),
                    ),
                )
            }))
            .build()
            .unwrap();

        let attacker_role = RoleBuilder::new("attacker")
            .min(1)
            .max(2)
            .score(BtCallback::new_native(|_| 75.0))
            .exclude(BtCallback::new_native(|s| s.player_id == PlayerId::new(3)))
            .behavior(BtCallback::new_native(|_| {
                super::super::rhai_types::RhaiBehaviorNode(
                    super::super::bt_node::BehaviorNode::Noop(
                        super::super::bt_node::NoopNode::new(),
                    ),
                )
            }))
            .build()
            .unwrap();

        let problem = RoleAssignmentProblem {
            roles: vec![
                goalkeeper_role,
                attacker_role,
                create_test_role("defender", 1, 2),
            ],
        };

        let assignments = solver
            .solve(&problem, &active_robots, &team_data, &engine)
            .unwrap();

        // Check constraints are satisfied
        assert_eq!(assignments[&PlayerId::new(0)], "goalkeeper");

        // Robot 3 should not be an attacker (excluded)
        assert_ne!(
            assignments.get(&PlayerId::new(3)),
            Some(&"attacker".to_string())
        );
    }

    #[test]
    fn test_role_builder_errors() {
        // Missing scorer
        let result = RoleBuilder::new("test")
            .behavior(BtCallback::new_native(|_| {
                super::super::rhai_types::RhaiBehaviorNode(
                    super::super::bt_node::BehaviorNode::Noop(
                        super::super::bt_node::NoopNode::new(),
                    ),
                )
            }))
            .build();
        assert!(result.is_err());

        // Missing behavior
        let result = RoleBuilder::new("test")
            .score(BtCallback::new_native(|_| 50.0))
            .build();
        assert!(result.is_err());
    }
}
