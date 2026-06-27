//! Role definitions + the greedy hysteretic assignment solver, ported from the
//! old engine. The solver builds a robot×role cost matrix from each role's
//! scorer/filters and runs the `linear_assignment` Hungarian-style solver.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use dies_strategy_protocol::{PlayerId, SkillStatus, WorldSnapshot};
use linear_assignment::MatrixSize;
use nalgebra::DMatrix;

use super::nodes::BehaviorNode;
use super::situation::{BtContext, RobotSituation};
use super::BtCallback;

/// A role the strategy wants filled, with its scorer, filters, and tree builder.
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
            "Role {{ name: {}, min: {}, max: {} }}",
            self.name, self.min_count, self.max_count
        )
    }
}

#[derive(Clone, Debug)]
pub struct RoleAssignmentProblem {
    pub roles: Vec<Role>,
}

/// Fluent builder for a [`Role`].
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
    pub fn new(name: impl Into<String>, index: usize) -> Self {
        Self {
            name: name.into(),
            index,
            min_count: 0,
            max_count: 6,
            can_be_reassigned: true,
            if_must_reassign_can_we_do_it_now: false,
            scorer: None,
            require_filter: None,
            exclude_filter: None,
            tree_builder: None,
        }
    }

    pub fn min(&mut self, count: usize) -> &mut Self {
        self.min_count = count;
        self
    }

    pub fn max(&mut self, count: usize) -> &mut Self {
        self.max_count = count;
        self
    }

    pub fn count(&mut self, count: usize) -> &mut Self {
        self.min_count = count;
        self.max_count = count;
        self
    }

    pub fn can_be_reassigned(&mut self, can_be_reassigned: bool) -> &mut Self {
        self.can_be_reassigned = can_be_reassigned;
        self
    }

    pub fn if_must_reassign_can_we_do_it_now(&mut self, v: bool) -> &mut Self {
        self.if_must_reassign_can_we_do_it_now = v;
        self
    }

    pub fn score(&mut self, scorer: impl BtCallback<f64>) -> &mut Self {
        self.scorer = Some(Arc::new(scorer));
        self
    }

    pub fn require(&mut self, filter: impl BtCallback<bool>) -> &mut Self {
        self.require_filter = Some(Arc::new(filter));
        self
    }

    pub fn exclude(&mut self, filter: impl BtCallback<bool>) -> &mut Self {
        self.exclude_filter = Some(Arc::new(filter));
        self
    }

    pub fn behavior(&mut self, builder: impl BtCallback<BehaviorNode>) -> &mut Self {
        self.tree_builder = Some(Arc::new(builder));
        self
    }

    pub fn build(self) -> Result<Role> {
        let scorer = self
            .scorer
            .ok_or_else(|| anyhow!("Role '{}' missing scorer", self.name))?;
        let tree_builder = self
            .tree_builder
            .ok_or_else(|| anyhow!("Role '{}' missing behavior tree", self.name))?;
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

/// Greedy assignment solver with role-keeping hysteresis.
pub struct RoleAssignmentSolver {
    hysteresis_bonus: f64,
}

impl Default for RoleAssignmentSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl RoleAssignmentSolver {
    pub fn new() -> Self {
        Self {
            hysteresis_bonus: 40.0,
        }
    }

    pub fn solve(
        &mut self,
        problem: &RoleAssignmentProblem,
        active_robots: &[PlayerId],
        world: Arc<WorldSnapshot>,
        previous_assignments: Option<&HashMap<PlayerId, String>>,
    ) -> (HashMap<PlayerId, String>, Vec<String>) {
        if active_robots.is_empty() {
            return (HashMap::new(), Vec::new());
        }

        // Expand role slots: first the required (min) slots, then optional ones up
        // to max, in declaration order.
        let mut roles = Vec::<Role>::new();
        'outer: for role in problem.roles.iter() {
            for _ in 0..role.min_count {
                roles.push(role.clone());
                if roles.len() == active_robots.len() {
                    break 'outer;
                }
            }
        }
        for role in problem.roles.iter() {
            for _ in role.min_count..role.max_count {
                roles.push(role.clone());
            }
        }

        if roles.is_empty() {
            return (HashMap::new(), Vec::new());
        }

        let prev_arc: Arc<HashMap<PlayerId, String>> =
            Arc::new(previous_assignments.cloned().unwrap_or_default());

        let mut score_matrix = DMatrix::zeros(roles.len(), active_robots.len());
        for (i, robot) in active_robots.iter().enumerate() {
            for (j, role) in roles.iter().enumerate() {
                let score = self.get_score(*robot, role, world.clone(), &prev_arc);
                let discount_factor = (role.index + 1) * j; // prefer min-count + earlier roles
                score_matrix[(j, i)] = score - (discount_factor as i64);
            }
        }

        let active_robots: Vec<PlayerId> = if active_robots.len() > roles.len() {
            active_robots.iter().take(roles.len()).copied().collect()
        } else {
            active_robots.to_vec()
        };

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

        (
            assignments,
            result.iter().map(|(_, j)| roles[*j].name.clone()).collect(),
        )
    }

    fn get_score(
        &self,
        robot_id: PlayerId,
        role: &Role,
        world: Arc<WorldSnapshot>,
        previous_assignments: &Arc<HashMap<PlayerId, String>>,
    ) -> i64 {
        let situation = RobotSituation::new(
            robot_id,
            world,
            BtContext::new(),
            previous_assignments.clone(),
            SkillStatus::Idle,
        );
        let is_eligible = !violates_filters(role, &situation);
        let mut score = if is_eligible {
            -(role.scorer)(&situation) as i64
        } else {
            1_000_000
        };

        if let Some(prev_role) = previous_assignments.get(&robot_id) {
            if prev_role == &role.name {
                score -= self.hysteresis_bonus as i64;
            }
        }

        score
    }
}

fn violates_filters(role: &Role, situation: &RobotSituation) -> bool {
    if let Some(require) = &role.require_filter {
        if !require(situation) {
            return true;
        }
    }
    if let Some(exclude) = &role.exclude_filter {
        if exclude(situation) {
            return true;
        }
    }
    false
}
