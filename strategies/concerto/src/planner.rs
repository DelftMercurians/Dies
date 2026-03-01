use dies_strategy_api::prelude::*;

use crate::geometry;
use crate::possession::PossessionState;

/// A single step in a plan.
#[derive(Debug, Clone)]
pub enum Waypoint {
    /// Go to the ball and capture it.
    Capture { robot_id: PlayerId },
    /// Dribble the ball toward a target area.
    Dribble { target_area: Vector2 },
    /// Pass the ball to a target area.
    Pass { target_area: Vector2 },
    /// Shoot the ball at a specific target (typically the goal).
    Shoot { target: Vector2 },
}

/// A sequence of waypoints assigned to an active robot.
#[derive(Debug, Clone)]
pub struct Plan {
    pub waypoints: Vec<Waypoint>,
    pub active_robot: PlayerId,
}

/// The Planner selects a plan based on the current world state and possession.
pub struct Planner {
    current_plan: Option<Plan>,
}

impl Planner {
    /// Create a new Planner with no initial plan.
    pub fn new() -> Self {
        Self {
            current_plan: None,
        }
    }

    /// Returns a reference to the current plan, if any.
    pub fn current_plan(&self) -> Option<&Plan> {
        self.current_plan.as_ref()
    }

    /// Re-evaluate the situation and produce a new plan.
    ///
    /// Returns `None` when the ball is not visible (nothing to plan around).
    pub fn replan(&mut self, world: &World, possession: &PossessionState) -> Option<&Plan> {
        let ball_pos = world.ball_position()?;
        let goal_center = world.opp_goal_center();

        let plan = match possession {
            // ── Branch 1 & 2: We have the ball ──────────────────────────
            PossessionState::We(robot_id) => {
                let robot_id = *robot_id;
                let carrier = world.own_player(robot_id)?;
                let carrier_pos = carrier.position;

                let clear = geometry::is_clear_shot(
                    carrier_pos,
                    goal_center,
                    world.opp_players(),
                    400.0,
                );
                let close_enough = (carrier_pos - goal_center).norm() < 3500.0;

                if clear && close_enough {
                    // Branch 1: clear shot — just shoot.
                    Plan {
                        waypoints: vec![Waypoint::Shoot { target: goal_center }],
                        active_robot: robot_id,
                    }
                } else {
                    // Branch 2: no clear shot — pass or dribble forward, then shoot.
                    let field_half_length = world.field_length() / 2.0;
                    let field_half_width = world.field_width() / 2.0;

                    let first_waypoint = if let Some(pass_area) = geometry::best_pass_area(
                        carrier_pos,
                        world.opp_players(),
                        field_half_length,
                        field_half_width,
                    ) {
                        Waypoint::Pass {
                            target_area: pass_area,
                        }
                    } else {
                        // No good pass — dribble toward opponent goal, clamped to field.
                        let raw_target = carrier_pos + Vector2::new(1500.0, 0.0);
                        let clamped = Vector2::new(
                            raw_target.x.clamp(-field_half_length, field_half_length),
                            raw_target.y.clamp(-field_half_width, field_half_width),
                        );
                        Waypoint::Dribble {
                            target_area: clamped,
                        }
                    };

                    Plan {
                        waypoints: vec![first_waypoint, Waypoint::Shoot { target: goal_center }],
                        active_robot: robot_id,
                    }
                }
            }

            // ── Branch 3: Ball is loose ─────────────────────────────────
            PossessionState::Loose => {
                let keeper_id = world.our_keeper_id();

                // Prefer non-keeper players; fall back to any player.
                let closest = world
                    .own_players()
                    .iter()
                    .filter(|p| Some(p.id) != keeper_id)
                    .min_by(|a, b| {
                        let da = (a.position - ball_pos).norm();
                        let db = (b.position - ball_pos).norm();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .or_else(|| world.closest_own_player_to(ball_pos))?;

                Plan {
                    waypoints: vec![Waypoint::Capture {
                        robot_id: closest.id,
                    }],
                    active_robot: closest.id,
                }
            }

            // ── Branch 4: Opponent has the ball ─────────────────────────
            PossessionState::Opponent(opp_id) => {
                let opp = world.opp_player(*opp_id)?;
                let opp_pos = opp.position;
                let own_goal = world.own_goal_center();
                let keeper_id = world.our_keeper_id();

                // Candidates: own players excluding the goalkeeper.
                let candidates: Vec<&PlayerState> = world
                    .own_players()
                    .iter()
                    .filter(|p| Some(p.id) != keeper_id)
                    .collect();

                // Prefer a player that is between the opponent and our goal.
                let between_players: Vec<&&PlayerState> = candidates
                    .iter()
                    .filter(|p| geometry::is_between(p.position, opp_pos, own_goal))
                    .collect();

                let interceptor = if !between_players.is_empty() {
                    // Among "between" players, pick the closest to the opponent.
                    between_players
                        .into_iter()
                        .min_by(|a, b| {
                            let da = (a.position - opp_pos).norm();
                            let db = (b.position - opp_pos).norm();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .copied()
                } else if !candidates.is_empty() {
                    // No player between — pick the closest to the opponent.
                    candidates
                        .iter()
                        .min_by(|a, b| {
                            let da = (a.position - opp_pos).norm();
                            let db = (b.position - opp_pos).norm();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .copied()
                } else {
                    // All players are keepers (edge case) — use closest overall.
                    world.closest_own_player_to(opp_pos)
                };

                let interceptor = interceptor?;

                Plan {
                    waypoints: vec![Waypoint::Capture {
                        robot_id: interceptor.id,
                    }],
                    active_robot: interceptor.id,
                }
            }
        };

        self.current_plan = Some(plan);
        self.current_plan.as_ref()
    }
}
