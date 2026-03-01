use std::collections::HashMap;

use dies_strategy_api::prelude::*;

use crate::geometry;

/// A formation role: a named position with an importance score.
#[derive(Debug, Clone)]
pub struct Role {
    pub name: &'static str,
    pub position: Vector2,
    pub importance: f64,
}

/// Stateless formation system. Recomputes roles and assignments every tick.
pub struct Formation;

impl Default for Formation {
    fn default() -> Self {
        Self::new()
    }
}

impl Formation {
    pub fn new() -> Self {
        Formation
    }

    /// Compute the set of available roles for this tick.
    pub fn compute_roles(
        &self,
        world: &World,
        plan_context: Option<Vector2>,
        field_half_length: f64,
        field_half_width: f64,
    ) -> Vec<Role> {
        let mut roles = Vec::with_capacity(8);

        let ball_pos = world.ball_position().unwrap_or(Vector2::new(0.0, 0.0));
        let own_goal = world.own_goal_center();

        // 1. Shadow roles (2) — defensive coverage between ball and our goal
        let (shadow_l, shadow_r) =
            geometry::compute_shadow_positions(ball_pos, own_goal, 1500.0, 400.0);
        let shadow_importance = if ball_pos.x < 0.0 { 10.0 } else { 5.0 };
        roles.push(Role {
            name: "Shadow-L",
            position: shadow_l,
            importance: shadow_importance,
        });
        roles.push(Role {
            name: "Shadow-R",
            position: shadow_r,
            importance: shadow_importance,
        });

        // 2. Plan context role (0 or 1) — where the planner wants a receiver
        if let Some(area) = plan_context {
            roles.push(Role {
                name: "PlanCtx",
                position: area,
                importance: 8.0,
            });
        }

        // 3. Offensive support roles (2) — forward positions nudged away from opponents
        let mut support_l = Vector2::new(field_half_length * 0.5, field_half_width * 0.3);
        let mut support_r = Vector2::new(field_half_length * 0.5, -field_half_width * 0.3);

        // Nudge away from nearest opponent if too close
        for pos in [&mut support_l, &mut support_r] {
            if let Some(closest_opp) = closest_opponent_to(*pos, world.opp_players()) {
                let dist = (closest_opp - *pos).norm();
                if dist < 800.0 && dist > 1e-6 {
                    let away = (*pos - closest_opp).normalize();
                    *pos += away * 400.0;
                }
            }
        }

        roles.push(Role {
            name: "Support-L",
            position: support_l,
            importance: 3.0,
        });
        roles.push(Role {
            name: "Support-R",
            position: support_r,
            importance: 3.0,
        });

        // 4. Fallback spread roles (2) — midfield coverage
        roles.push(Role {
            name: "Spread-L",
            position: Vector2::new(0.0, field_half_width * 0.2),
            importance: 1.0,
        });
        roles.push(Role {
            name: "Spread-R",
            position: Vector2::new(0.0, -field_half_width * 0.2),
            importance: 1.0,
        });

        roles
    }

    /// Greedy assignment: assign robots to roles by importance, picking the closest
    /// unassigned robot for each role.
    pub fn assign_roles(
        &self,
        robot_ids: &[PlayerId],
        roles: &[Role],
        world: &World,
    ) -> HashMap<PlayerId, Role> {
        let mut assignments = HashMap::new();
        let mut available: Vec<PlayerId> = robot_ids.to_vec();

        // Sort roles by importance descending
        let mut sorted_indices: Vec<usize> = (0..roles.len()).collect();
        sorted_indices.sort_by(|a, b| {
            roles[*b]
                .importance
                .partial_cmp(&roles[*a].importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for idx in sorted_indices {
            if available.is_empty() {
                break;
            }
            let role = &roles[idx];

            // Find closest unassigned robot
            let mut best_i = 0;
            let mut best_dist = f64::MAX;
            for (i, id) in available.iter().enumerate() {
                if let Some(player) = world.own_player(*id) {
                    let dist = (player.position - role.position).norm();
                    if dist < best_dist {
                        best_dist = dist;
                        best_i = i;
                    }
                }
            }

            let chosen_id = available.remove(best_i);
            assignments.insert(chosen_id, role.clone());
        }

        assignments
    }

    /// Compute the goalkeeper's position: on the goal line, tracking the ball.
    pub fn compute_goalkeeper_position(
        &self,
        ball_pos: Vector2,
        field_half_length: f64,
        goal_width: f64,
    ) -> Vector2 {
        let x = -field_half_length + 200.0;
        let y = (ball_pos.y * 0.5).clamp(-goal_width / 2.0, goal_width / 2.0);
        Vector2::new(x, y)
    }
}

/// Find the position of the closest opponent to a given point.
fn closest_opponent_to(pos: Vector2, opponents: &[PlayerState]) -> Option<Vector2> {
    opponents
        .iter()
        .min_by(|a, b| {
            let da = (a.position - pos).norm();
            let db = (b.position - pos).norm();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.position)
}
