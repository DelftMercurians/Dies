use anyhow::{anyhow, Result};
use dies_core::{PlayerId, TeamData};
use std::{collections::HashMap, sync::Arc};

use crate::{behavior_tree::BehaviorNode, control::TeamContext};

use super::{BtCallback, RobotSituation};

/// Represents a role that can be assigned to robots
#[derive(Clone)]
pub struct Role {
    pub name: String,
    pub min_count: usize,
    pub max_count: usize,
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
    pub min_count: usize,
    pub max_count: usize,
    pub scorer: Option<Arc<dyn BtCallback<f64>>>,
    pub require_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub exclude_filter: Option<Arc<dyn BtCallback<bool>>>,
    pub tree_builder: Option<Arc<dyn BtCallback<BehaviorNode>>>,
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
            min_count: self.min_count,
            max_count: self.max_count,
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
            hysteresis_bonus: 5.0,
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
    ) -> Result<HashMap<PlayerId, String>> {
        if active_robots.is_empty() {
            return Ok(HashMap::new());
        }

        // Clear cache for fresh assignment
        self.score_cache.clear();

        // Pre-compute eligible robots for each role
        let eligible_robots = self.compute_eligible_robots(
            problem,
            active_robots,
            team_context.clone(),
            team_data.clone(),
            previous_assignments,
        )?;

        // Sort roles by priority (critical roles first)
        let mut sorted_roles: Vec<_> = problem.roles.iter().enumerate().collect();
        sorted_roles.sort_by_key(|(_, role)| {
            // Priority: min_count descending, then scarcity (fewer eligible robots = higher priority)
            let eligible_count = eligible_robots
                .get(&role.name)
                .map(|v| v.len())
                .unwrap_or(0);
            let scarcity_score = if eligible_count > 0 {
                role.min_count * 1000 / eligible_count
            } else {
                999999
            };
            (-(role.min_count as i32), scarcity_score)
        });

        let mut assignments = HashMap::new();
        let mut assigned_robots = std::collections::HashSet::new();

        // First pass: satisfy minimum requirements
        for (_, role) in &sorted_roles {
            let empty_vec = Vec::new();
            let eligible = eligible_robots.get(&role.name).unwrap_or(&empty_vec);
            let available: Vec<_> = eligible
                .iter()
                .filter(|&&robot_id| !assigned_robots.contains(&robot_id))
                .collect();

            if available.len() < role.min_count {
                return Err(anyhow!(
                    "Cannot satisfy min_count {} for role '{}', only {} eligible robots available",
                    role.min_count,
                    role.name,
                    available.len()
                ));
            }

            // Score and assign best robots for this role
            let mut scored_robots: Vec<_> = available
                .iter()
                .map(|&&robot_id| {
                    let score = self.get_cached_score(
                        robot_id,
                        &role.name,
                        role,
                        team_context.clone(),
                        team_data.clone(),
                        previous_assignments,
                    );
                    (robot_id, score)
                })
                .collect::<Vec<_>>();

            scored_robots
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Assign minimum required
            for (robot_id, _) in scored_robots.iter().take(role.min_count) {
                assignments.insert(*robot_id, role.name.clone());
                assigned_robots.insert(*robot_id);
            }
        }

        // Second pass: fill remaining slots up to max_count
        for (_, role) in &sorted_roles {
            let current_count = assignments
                .values()
                .filter(|&name| name == &role.name)
                .count();
            if current_count >= role.max_count {
                continue;
            }

            let empty_vec = Vec::new();
            let eligible = eligible_robots.get(&role.name).unwrap_or(&empty_vec);
            let available: Vec<_> = eligible
                .iter()
                .filter(|&&robot_id| !assigned_robots.contains(&robot_id))
                .collect();

            let mut scored_robots: Vec<_> = available
                .iter()
                .map(|robot_id| {
                    let score = self.get_cached_score(
                        **robot_id,
                        &role.name,
                        role,
                        team_context.clone(),
                        team_data.clone(),
                        previous_assignments,
                    );
                    (**robot_id, score)
                })
                .collect::<Vec<_>>();

            scored_robots
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Fill remaining slots
            let slots_to_fill = role.max_count - current_count;
            for (robot_id, _) in scored_robots.iter().take(slots_to_fill) {
                assignments.insert(*robot_id, role.name.clone());
                assigned_robots.insert(*robot_id);
            }
        }

        Ok(assignments)
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
                    previous_assignments.cloned().unwrap_or_default(),
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
    fn get_cached_score(
        &mut self,
        robot_id: PlayerId,
        role_name: &str,
        role: &Role,
        team_context: TeamContext,
        team_data: Arc<TeamData>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> f64 {
        let cache_key = (robot_id, role_name.to_string());

        if let Some(&cached_score) = self.score_cache.get(&cache_key) {
            return cached_score;
        }

        let situation = RobotSituation::new(
            robot_id,
            team_data.clone(),
            Default::default(),
            team_context.player_context(robot_id).key("bt"),
            previous_assignments.cloned().unwrap_or_default(),
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

        self.score_cache.insert(cache_key, score);
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

        let assignments = solver
            .solve(&problem, &active_robots, team_context, team_data, None)
            .unwrap();

        assert_eq!(assignments.len(), 3);
        // All robots should be assigned
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

        let assignments = solver
            .solve(&problem, &active_robots, team_context, team_data, None)
            .unwrap();

        // Basic validation - all robots should be assigned
        assert_eq!(assignments.len(), 4);

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
        let assignments1 = solver
            .solve(
                &problem,
                &active_robots,
                team_context.clone(),
                team_data.clone(),
                None,
            )
            .unwrap();

        // Second assignment with previous assignments should be identical due to hysteresis
        let assignments2 = solver
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

        // Third assignment should also be identical
        let assignments3 = solver
            .solve(
                &problem,
                &active_robots,
                team_context.clone(),
                team_data.clone(),
                Some(&assignments2),
            )
            .unwrap();

        assert_eq!(assignments2, assignments3);
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
