use anyhow::{anyhow, Result};
use dies_core::{PlayerId, TeamData};
use std::{collections::HashMap, sync::Arc};
use std::time::Instant;

use crate::{behavior_tree::BehaviorNode, control::TeamContext};

use super::{BtCallback, RobotSituation};
use linear_assignment::MatrixSize;
use nalgebra::DMatrix;

/// Represents a role that can be assigned to robots
#[derive(Clone)]
pub struct Role {
    pub name: String,
    pub index: usize,
    pub min_count: usize,
    pub max_count: usize,
    pub can_be_reassigned: bool,
    pub if_must_reassign_can_we_do_it_now: bool,
    pub scorer: Arc<dyn BtCallback<f64>>,
    pub require_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub exclude_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub tree_builder: Arc<dyn BtCallback<BehaviorNode>>,
}

impl std::fmt::Debug for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Role {{ name: {}, min_count: {}, max_count: {} }}",
            self.name, self.min_count, self.max_count,
        )
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
    pub name: String,
    pub index: usize,
    pub min_count: usize,
    pub max_count: usize,
    pub can_be_reassigned: bool,
    pub if_must_reassign_can_we_do_it_now: bool,
    pub scorer: Option<Arc<dyn BtCallback<f64>>>,
    pub require_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub exclude_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub tree_builder: Option<Arc<dyn BtCallback<BehaviorNode>>>,
}

impl RoleBuilder {
    /// Create a new role builder
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            index: index,
            min_count: 0,
            max_count: 6,
            can_be_reassigned: true,
            if_must_reassign_can_we_do_it_now: false,
            scorer: None,
            exclude_filter: None,
            require_filter: None,
            tree_builder: None,
        }
    }

    /// Set minimum number of robots for this role
    pub fn min(&mut self, count: usize) -> &mut Self {
        self.min_count = count;
        self
    }

    /// Set maximum number of robots for this role
    pub fn max(&mut self, count: usize) -> &mut Self {
        self.max_count = count;
        self
    }

    /// Set exact count (min = max)
    pub fn count(&mut self, count: usize) -> &mut Self {
        self.min_count = count;
        self.max_count = count;
        self
    }

    /// Set if the role can be reassigned
    pub fn can_be_reassigned(&mut self, can_be_reassigned: bool) -> &mut Self {
        self.can_be_reassigned = can_be_reassigned;
        self
    }

    /// Set if the role can be reassigned
    pub fn if_must_reassign_can_we_do_it_now(
        &mut self,
        if_must_reassign_can_we_do_it_now: bool,
    ) -> &mut Self {
        self.if_must_reassign_can_we_do_it_now = if_must_reassign_can_we_do_it_now;
        self
    }

    /// Set the scoring function for this role
    pub fn score(&mut self, scorer: impl BtCallback<f64>) -> &mut Self {
        self.scorer = Some(Arc::new(scorer));
        self
    }

    /// Set a filter function that must return true for a robot to be considered for this role
    pub fn require(&mut self, filter: impl BtCallback<bool>) -> &mut Self {
        self.require_filter = Some(Arc::new(filter));
        self
    }

    /// Set a filter function that must return false for a robot to be considered for this role
    pub fn exclude(&mut self, filter: impl BtCallback<bool>) -> &mut Self {
        self.exclude_filter = Some(Arc::new(filter));
        self
    }

    /// Set the behavior tree builder
    pub fn behavior(&mut self, builder: impl BtCallback<BehaviorNode>) -> &mut Self {
        self.tree_builder = Some(Arc::new(builder));
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
            index: self.index,
            min_count: self.min_count,
            max_count: self.max_count,
            can_be_reassigned: self.can_be_reassigned,
            if_must_reassign_can_we_do_it_now: self.if_must_reassign_can_we_do_it_now,
            scorer,
            require_filter: self.require_filter,
            exclude_filter: self.exclude_filter,
            tree_builder,
        })
    }
}

/// Solver for the role assignment problem using a fast greedy algorithm
pub struct RoleAssignmentSolver {
    // Cache for performance
    score_cache: HashMap<(PlayerId, String), f64>,
    /// Bonus score added to robots that already have the role (hysteresis)
    hysteresis_bonus: f64,
}

impl RoleAssignmentSolver {
    pub fn new() -> Self {
        Self {
            score_cache: HashMap::new(),
            hysteresis_bonus: 20.0,
        }
    }

    /// Solve the role assignment problem using fast greedy algorithm with hysteresis
    pub fn solve(
        &mut self,
        problem: &RoleAssignmentProblem,
        active_robots: &[PlayerId],
        team_context: TeamContext,
        team_data: Arc<TeamData>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> Result<(HashMap<PlayerId, String>, Vec<String>)> {
        if active_robots.is_empty() {
            return Ok((HashMap::new(), Vec::new()));
        }

        let start_time = Instant::now();

        let mut roles = Vec::<Role>::new();
        // generate required roles
        'outer: for role in problem.roles.iter() {
            for i in 0..role.min_count {
                roles.push(role.clone());

                if roles.len() == active_robots.len() {
                    break 'outer;
                }
            }
        }

        // generate roles up to max
        'outer: for role in problem.roles.iter() {
            for i in role.min_count..role.max_count {
                roles.push(role.clone());

                if roles.len() == active_robots.len() {
                    // break 'outer; - not necessary if there is weighting
                }
            }
        }

        // each role is duplicated according to max count
        let mut score_matrix = DMatrix::zeros(roles.len(), active_robots.len());
        for (i, robot) in active_robots.iter().enumerate() {
            for (j, role) in roles.iter().enumerate() {
                let score = self.get_score(
                    *robot,
                    &role.name,
                    role,
                    team_context.clone(),
                    team_data.clone(),
                    previous_assignments,
                );
                let discount_factor = (role.index + 1) * j; // prefer closer to min and earlier in
                                                            // sequence roles
                score_matrix[(j, i)] = (score * (discount_factor as f64)) as i64;
            }
        }

        let result = linear_assignment::solver(
            &mut score_matrix.transpose(),
            &MatrixSize {
                columns: roles.len(),
                rows: active_robots.len(),
            },
        );

        let mut assignments = HashMap::new();
        for (i, j) in result.iter() {
            assignments.insert(active_robots[*i], roles[*j].name.clone());
        }
        team_context.debug_value(
            "role_assignment_time",
            start_time.elapsed().as_micros() as f64,
        );

        Ok((
            assignments,
            result.iter().map(|(_, j)| roles[*j].name.clone()).collect(),
        ))
    }

    /// Compute eligible robots for each role (applying filters)
    fn compute_eligible_robots(
        &mut self,
        problem: &RoleAssignmentProblem,
        robots: &[PlayerId],
        team_context: TeamContext,
        team_data: Arc<TeamData>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> Result<HashMap<String, Vec<PlayerId>>> {
        let mut eligible_robots = HashMap::new();

        for role in &problem.roles {
            let mut eligible = Vec::new();

            for &robot_id in robots {
                let situation = RobotSituation::new(
                    robot_id,
                    team_data.clone(),
                    Default::default(),
                    team_context.player_context(robot_id).key("bt"),
                    previous_assignments.cloned().unwrap_or_default().into(),
                    team_context.team_color(),
                );

                if !self.violates_filters(&role, &situation) {
                    eligible.push(robot_id);
                }
            }

            eligible_robots.insert(role.name.clone(), eligible);
        }

        Ok(eligible_robots)
    }

    /// Get cached score for robot-role combination
    fn get_score(
        &mut self,
        robot_id: PlayerId,
        role_name: &str,
        role: &Role,
        team_context: TeamContext,
        team_data: Arc<TeamData>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> f64 {
        let situation = RobotSituation::new(
            robot_id,
            team_data.clone(),
            Default::default(),
            team_context.player_context(robot_id).key("bt"),
            previous_assignments.cloned().unwrap_or_default().into(),
            team_context.team_color(),
        );
        let mut score = (role.scorer)(&situation);

        // Apply hysteresis bonus - strongly prefer keeping current role assignment
        if let Some(prev_assignments) = previous_assignments {
            if let Some(prev_role) = prev_assignments.get(&robot_id) {
                if prev_role == role_name {
                    score *= self.hysteresis_bonus;
                }
            }
        }

        score
    }

    /// Check if a robot violates role filters
    fn violates_filters(&self, role: &Role, situation: &RobotSituation) -> bool {
        // Check require filter - must return true
        if let Some(ref require_filter) = role.require_filter {
            if !require_filter(situation) {
                return true;
            }
        }

        // Check exclude filter - must return false
        if let Some(ref exclude_filter) = role.exclude_filter {
            if exclude_filter(situation) {
                return true;
            }
        }

        false
    }
}

impl Default for RoleAssignmentSolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::behavior_tree::NoopNode;

    use super::*;
    use dies_core::{
        Angle, BallData, GameState, GameStateData, PlayerData, SideAssignment, SysStatus,
        TeamColor, TeamData, Vector2, Vector3,
    };

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
                handicaps: HashSet::new(),
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
                yellow_cards: 0,
                freekick_kicker: None,
                max_allowed_bots: 3,
                our_keeper_id: None,
            },
            field_geom: None,
            ball_on_our_side: None,
            ball_on_opp_side: None,
            kicked_ball: None,
        }
    }

    fn create_test_role(name: &str, min: usize, max: usize) -> Role {
        let mut role = RoleBuilder::new(name);
        role.min(min)
            .max(max)
            .score(|_| 50.0)
            .behavior(|_| BehaviorNode::Noop(NoopNode::new()));
        role.build().unwrap()
    }

    #[test]
    fn test_role_builder() {
        let mut role = RoleBuilder::new("attacker");
        role.min(1)
            .min(1)
            .max(2)
            .score(|_| 100.0)
            .require(|_| true)
            .exclude(|_| false)
            .behavior(|_| BehaviorNode::Noop(NoopNode::new()));
        let role = role.build().unwrap();

        assert_eq!(role.name, "attacker");
        assert_eq!(role.min_count, 1);
        assert_eq!(role.max_count, 2);
        assert!(role.require_filter.is_some());
        assert!(role.exclude_filter.is_some());
    }

    #[test]
    fn test_simple_assignment() {
        let mut solver = RoleAssignmentSolver::new();
        let team_context = TeamContext::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        let team_data = Arc::new(create_test_team_data(3));
        let active_robots = vec![PlayerId::new(0), PlayerId::new(1), PlayerId::new(2)];

        let problem = RoleAssignmentProblem {
            roles: vec![
                create_test_role("goalkeeper", 1, 1),
                create_test_role("attacker", 1, 1),
                create_test_role("defender", 1, 1),
            ],
        };

        let (assignments, priority_list) = solver
            .solve(&problem, &active_robots, team_context, team_data, None)
            .unwrap();

        assert_eq!(assignments.len(), 3);
        // All robots should be assigned
        assert_eq!(priority_list.len(), 3);
        // Priority list should contain all role names
        assert!(priority_list.contains(&"goalkeeper".to_string()));
        assert!(priority_list.contains(&"attacker".to_string()));
        assert!(priority_list.contains(&"defender".to_string()));
    }

    #[test]
    fn test_filter_constraints() {
        let mut solver = RoleAssignmentSolver::new();
        let team_context = TeamContext::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        let team_data = Arc::new(create_test_team_data(4));
        let active_robots = vec![
            PlayerId::new(0),
            PlayerId::new(1),
            PlayerId::new(2),
            PlayerId::new(3),
        ];

        // Create simpler roles without complex filters for now
        let problem = RoleAssignmentProblem {
            roles: vec![
                create_test_role("goalkeeper", 1, 1),
                create_test_role("attacker", 1, 2),
                create_test_role("defender", 1, 2),
            ],
        };

        let (assignments, priority_list) = solver
            .solve(&problem, &active_robots, team_context, team_data, None)
            .unwrap();

        // Basic validation - all robots should be assigned
        assert_eq!(assignments.len(), 4);
        assert_eq!(priority_list.len(), 3);

        // Count by role
        let mut role_counts = std::collections::HashMap::new();
        for role_name in assignments.values() {
            *role_counts.entry(role_name.as_str()).or_insert(0) += 1;
        }

        assert_eq!(role_counts.get("goalkeeper"), Some(&1));
        assert!(role_counts
            .get("attacker")
            .map_or(false, |&c| c >= 1 && c <= 2));
        assert!(role_counts
            .get("defender")
            .map_or(false, |&c| c >= 1 && c <= 2));
    }

    #[test]
    fn test_hysteresis_prevents_oscillations() {
        let mut solver = RoleAssignmentSolver::new();
        let team_context = TeamContext::new(TeamColor::Blue, SideAssignment::YellowOnPositive);
        let team_data = Arc::new(create_test_team_data(3));
        let active_robots = vec![PlayerId::new(0), PlayerId::new(1), PlayerId::new(2)];

        let problem = RoleAssignmentProblem {
            roles: vec![
                create_test_role("goalkeeper", 1, 1),
                create_test_role("attacker", 1, 1),
                create_test_role("defender", 1, 1),
            ],
        };

        // First assignment (no previous assignments)
        let (assignments1, priority_list1) = solver
            .solve(
                &problem,
                &active_robots,
                team_context.clone(),
                team_data.clone(),
                None,
            )
            .unwrap();

        // Second assignment with previous assignments should be identical due to hysteresis
        let (assignments2, priority_list2) = solver
            .solve(
                &problem,
                &active_robots,
                team_context.clone(),
                team_data.clone(),
                Some(&assignments1),
            )
            .unwrap();

        // Should be identical assignments due to hysteresis
        assert_eq!(assignments1, assignments2);
        assert_eq!(priority_list1, priority_list2);

        // Third assignment should also be identical
        let (assignments3, priority_list3) = solver
            .solve(
                &problem,
                &active_robots,
                team_context.clone(),
                team_data.clone(),
                Some(&assignments2),
            )
            .unwrap();

        assert_eq!(assignments2, assignments3);
        assert_eq!(priority_list2, priority_list3);
    }

    #[test]
    fn test_role_builder_errors() {
        // Missing scorer
        let mut role = RoleBuilder::new("test");
        role.behavior(|_| BehaviorNode::Noop(NoopNode::new()));
        let result = role.build();
        assert!(result.is_err());

        // Missing behavior
        let mut role = RoleBuilder::new("test");
        role.score(|_| 50.0);
        let result = role.build();
        assert!(result.is_err());
    }
}
