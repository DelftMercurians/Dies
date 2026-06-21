//! Planner — deliberately dumb, event-driven, pass-ready.
//!
//! Given the stable possession, it picks at most one ball-state-transition
//! waypoint and the active robot. It runs only on discrete events (see `lib.rs`),
//! so plan continuity falls out for free. Stability of the active-robot choice
//! comes from physics (momentum-aware time-to-ball), not a stay-bonus.

use std::collections::HashMap;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::driver::FailReason;
use crate::geometry;
use crate::possession::Possession;

/// How a capture is performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureKind {
    /// Ball is loose — drive to it and pick it up.
    Loose,
    /// Ball is held by an opponent — challenge from a favourable angle.
    Steal { from: PlayerId },
}

/// A desired ball-state transition.
#[derive(Debug, Clone)]
pub enum Waypoint {
    Capture {
        kind: CaptureKind,
        robot: PlayerId,
    },
    Dribble {
        target_area: Vector2,
    },
    Shoot {
        target: Vector2,
    },
    /// Pass seam — defined for the future passing milestone, never emitted in v1.
    #[allow(dead_code)]
    Pass {
        passer: PlayerId,
        receiver_hint: Option<PlayerId>,
        target_area: Vector2,
    },
}

/// A plan: a sequence of waypoints driven by one active robot. v1 emits length-1;
/// replan-after-each makes any tail advisory.
#[derive(Debug, Clone)]
pub struct Plan {
    pub waypoints: Vec<Waypoint>,
    pub active_robot: PlayerId,
}

/// Inputs the planner needs beyond the world snapshot.
pub struct PlanInputs {
    pub keeper_id: Option<PlayerId>,
    pub double_touch_robot: Option<PlayerId>,
    pub now: f64,
}

/// Selects plans on events; remembers recent per-robot failures to avoid loops.
pub struct Planner {
    current_plan: Option<Plan>,
    recent_failures: HashMap<PlayerId, (FailReason, f64)>,
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner {
    pub fn new() -> Self {
        Self {
            current_plan: None,
            recent_failures: HashMap::new(),
        }
    }

    pub fn current_plan(&self) -> Option<&Plan> {
        self.current_plan.as_ref()
    }

    pub fn clear_plan(&mut self) {
        self.current_plan = None;
    }

    /// Record a waypoint failure so the next selection can avoid this robot
    /// briefly (used by the no-progress anti-loop in M3).
    pub fn record_failure(&mut self, robot: PlayerId, reason: FailReason, now: f64) {
        self.recent_failures.insert(robot, (reason, now));
    }

    /// Drop expired failure records; clear all on possession change.
    fn prune_failures(&mut self, now: f64) {
        self.recent_failures
            .retain(|_, (_, ts)| now - *ts < config::NOPROGRESS_TTL);
    }

    /// True if `id` recently failed with NoProgress and is still in the cooldown.
    fn recently_stuck(&self, id: PlayerId) -> bool {
        matches!(
            self.recent_failures.get(&id),
            Some((FailReason::NoProgress, _))
        )
    }

    /// Re-evaluate and produce a plan, or `None` to defer to Formation (defend).
    pub fn replan(
        &mut self,
        world: &World,
        possession: &Possession,
        inputs: &PlanInputs,
    ) -> Option<Plan> {
        self.prune_failures(inputs.now);

        let ball_pos = world.ball_position()?;
        let opp_goal = world.opp_goal_center();

        let plan = match *possession {
            // ── We have the ball ────────────────────────────────────────
            Possession::We(id) => {
                let carrier = world.own_player(id)?;
                let carrier_pos = carrier.position;

                let clear = geometry::is_clear_shot(
                    carrier_pos,
                    opp_goal,
                    world.opp_players(),
                    config::CLEAR_SHOT_CORRIDOR,
                );
                let in_range = (carrier_pos - opp_goal).norm() < config::SHOOT_RANGE;

                let waypoint = if clear && in_range {
                    Waypoint::Shoot { target: opp_goal }
                } else {
                    // *** PASS SEAM ***: future milestone inserts a Pass waypoint
                    // here (we-have-ball, no clear shot). v1 advances by dribbling.
                    Waypoint::Dribble {
                        target_area: self.advance_point(carrier_pos, opp_goal, world),
                    }
                };
                Plan {
                    waypoints: vec![waypoint],
                    active_robot: id,
                }
            }

            // ── Ball is loose ───────────────────────────────────────────
            Possession::Loose => {
                let robot = self.select_capturer(world, ball_pos, inputs)?;
                Plan {
                    waypoints: vec![Waypoint::Capture {
                        kind: CaptureKind::Loose,
                        robot,
                    }],
                    active_robot: robot,
                }
            }

            // ── Opponent has the ball ───────────────────────────────────
            Possession::Opp(oid) => {
                // M1 crude steal gate: only commit a challenger that is reasonably
                // close to the ball; otherwise defer to Formation. (M3 replaces this
                // with a proper conservative gate that protects deep defenders.)
                let robot = self.select_capturer(world, ball_pos, inputs)?;
                let robot_pos = world.own_player(robot)?.position;
                if (robot_pos - ball_pos).norm() > config::STEAL_MAX_DIST {
                    self.current_plan = None;
                    return None;
                }
                Plan {
                    waypoints: vec![Waypoint::Capture {
                        kind: CaptureKind::Steal { from: oid },
                        robot,
                    }],
                    active_robot: robot,
                }
            }
        };

        self.current_plan = Some(plan.clone());
        Some(plan)
    }

    /// Choose the robot that can reach the ball soonest (momentum-aware), excluding
    /// the keeper, the double-touch robot, and robots in a no-progress cooldown.
    fn select_capturer(
        &self,
        world: &World,
        ball_pos: Vector2,
        inputs: &PlanInputs,
    ) -> Option<PlayerId> {
        let pick = |relax_stuck: bool| -> Option<PlayerId> {
            world
                .own_players()
                .iter()
                .filter(|p| Some(p.id) != inputs.keeper_id)
                .filter(|p| Some(p.id) != inputs.double_touch_robot)
                .filter(|p| relax_stuck || !self.recently_stuck(p.id))
                .min_by(|a, b| {
                    let ta = geometry::redirect_time(
                        a.position,
                        a.velocity,
                        ball_pos,
                        config::V_MAX,
                        config::A_MAX,
                    );
                    let tb = geometry::redirect_time(
                        b.position,
                        b.velocity,
                        ball_pos,
                        config::V_MAX,
                        config::A_MAX,
                    );
                    ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|p| p.id)
        };

        // Prefer a robot not in cooldown; if everyone is stuck, allow reuse.
        pick(false).or_else(|| pick(true))
    }

    /// A point toward the opponent goal to dribble to, clamped to the field.
    fn advance_point(&self, from: Vector2, opp_goal: Vector2, world: &World) -> Vector2 {
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let dir = opp_goal - from;
        let n = dir.norm();
        let step = if n > 1e-6 {
            dir / n * config::DRIBBLE_ADVANCE
        } else {
            Vector2::new(config::DRIBBLE_ADVANCE, 0.0)
        };
        let raw = from + step;
        Vector2::new(
            raw.x.clamp(-half_len + 200.0, half_len - 200.0),
            raw.y.clamp(-half_wid + 200.0, half_wid - 200.0),
        )
    }
}
