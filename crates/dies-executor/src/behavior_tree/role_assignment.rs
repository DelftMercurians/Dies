use anyhow::{anyhow, Result};
use dies_core::{PlayerId, TeamData};
use std::time::Instant;
use std::{collections::HashMap, sync::Arc};

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
    /// Bonus score added to robots that already have the role (hysteresis)
    hysteresis_bonus: f64,
}

impl RoleAssignmentSolver {
    pub fn new() -> Self {
        Self {
            hysteresis_bonus: 40.0,
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
                score_matrix[(j, i)] = (score - (discount_factor as i64));
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
                    team_context.clone(),
                );

                if !self.violates_filters(&role, &situation) {
                    eligible.push(robot_id);
                }
            }

            eligible_robots.insert(role.name.clone(), eligible);
        }

        Ok(eligible_robots)
    }

    fn get_score(
        &mut self,
        robot_id: PlayerId,
        role_name: &str,
        role: &Role,
        team_context: TeamContext,
        team_data: Arc<TeamData>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> i64 {
        let situation = RobotSituation::new(
            robot_id,
            team_data.clone(),
            Default::default(),
            team_context.player_context(robot_id).key("bt"),
            previous_assignments.cloned().unwrap_or_default().into(),
            team_context.team_color(),
            team_context.clone(),
        );
        let is_eligible = !self.violates_filters(role, &situation);
        let mut score = if is_eligible {
            -(role.scorer)(&situation) as i64
        } else {
            1_000_000
        };

        // Apply hysteresis bonus - strongly prefer keeping current role assignment
        if let Some(prev_assignments) = previous_assignments {
            if let Some(prev_role) = prev_assignments.get(&robot_id) {
                if prev_role == role_name {
                    score -= self.hysteresis_bonus as i64;
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
                // println!(
                //     "{} violates require filter for {}",
                //     situation.player_id(),
                //     role.name
                // );
                return true;
            }
        }

        // Check exclude filter - must return false
        if let Some(ref exclude_filter) = role.exclude_filter {
            if exclude_filter(situation) {
                // println!(
                //     "{} violates exclude filter for {}",
                //     situation.player_id(),
                //     role.name
                // );
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
