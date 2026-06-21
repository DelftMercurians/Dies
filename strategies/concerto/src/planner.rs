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
    /// True during our in-play kickoff/free kick, where the kicker must release the
    /// ball forward (not dribble) to bring it into play and respect double-touch.
    pub our_attacking_restart: bool,
    /// Linear distance the ball has been carried from the contact point this
    /// possession (for the excessive-dribbling cap). 0 if we don't hold the ball.
    pub carried: f64,
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

                // Our attacking restart: this carrier is the kicker (it touched
                // first, or nobody has yet). It must release the ball forward — a
                // directed kick toward an open downfield zone — rather than dribble,
                // to bring the ball into play and avoid a double-touch. A teammate
                // then collects and normal play resumes. (Becomes a Pass in M4.)
                let is_kicker = inputs.our_attacking_restart
                    && (inputs.double_touch_robot.is_none()
                        || inputs.double_touch_robot == Some(id));

                let clear = geometry::is_clear_shot(
                    carrier_pos,
                    opp_goal,
                    world.opp_players(),
                    config::CLEAR_SHOT_CORRIDOR,
                );
                let in_range = (carrier_pos - opp_goal).norm() < config::SHOOT_RANGE;

                let waypoint = if is_kicker {
                    // Restart: always release forward (never dribble — double-touch).
                    // Kick at a supporter if one is well placed, else open space.
                    let target = self
                        .best_kickahead_target(world, id, inputs)
                        .unwrap_or_else(|| self.release_target(carrier_pos, opp_goal, world));
                    Waypoint::Shoot { target }
                } else if clear && in_range {
                    Waypoint::Shoot { target: opp_goal }
                } else {
                    // Primary advancement is a kick-ahead toward a supporter or open
                    // space (dribbling is unreliable and rule-capped). Dribble only as
                    // a small correction, and only while we're under the carry cap.
                    // *** PASS SEAM ***: M4 upgrades the supporter kick-ahead to a Pass.
                    match self.best_kickahead_target(world, id, inputs) {
                        Some(t) => Waypoint::Shoot { target: t },
                        None if inputs.carried < config::DRIBBLE_CORRECTION_LIMIT => {
                            Waypoint::Dribble {
                                target_area: self.correction_target(carrier_pos, opp_goal, world),
                            }
                        }
                        None => Waypoint::Shoot {
                            target: self.release_target(carrier_pos, opp_goal, world),
                        },
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

    /// Target for an attacking-restart release kick: the most open forward zone
    /// (toward a support teammate's area), falling back to the opponent half.
    fn release_target(&self, from: Vector2, opp_goal: Vector2, world: &World) -> Vector2 {
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        geometry::best_pass_area(from, world.opp_players(), half_len, half_wid).unwrap_or_else(
            || {
                // Fallback: a point well into the opponent half, biased toward goal.
                let dir = opp_goal - from;
                let n = dir.norm();
                let step = if n > 1e-6 {
                    dir / n * (half_len * 0.6)
                } else {
                    Vector2::new(half_len * 0.6, 0.0)
                };
                let raw = from + step;
                Vector2::new(
                    raw.x.clamp(-half_len + 200.0, half_len - 200.0),
                    raw.y.clamp(-half_wid + 200.0, half_wid - 200.0),
                )
            },
        )
    }

    /// Best kick-ahead target: a well-placed forward supporter (preferred) or open
    /// forward space toward goal. `None` only when congested (no supporter and no
    /// open area) — the rare case where a small corrective dribble is warranted.
    fn best_kickahead_target(
        &self,
        world: &World,
        carrier: PlayerId,
        inputs: &PlanInputs,
    ) -> Option<Vector2> {
        let carrier_pos = world.own_player(carrier)?.position;
        let opp_goal = world.opp_goal_center();
        let opps = world.opp_players();

        // 1. Forward supporter with an open lane from the ball. Score by how far
        //    forward (goalward) it is, weighted by lane openness.
        let best_supporter = world
            .own_players()
            .iter()
            .filter(|p| p.id != carrier)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| Some(p.id) != inputs.double_touch_robot)
            .filter(|p| p.position.x > carrier_pos.x + config::SUPPORTER_FWD_MARGIN)
            .filter_map(|p| {
                let open = geometry::lane_openness(
                    carrier_pos,
                    p.position,
                    opps,
                    config::KICK_LANE_CORRIDOR,
                );
                (open >= config::SUPPORTER_MIN_OPENNESS).then_some((p, open))
            })
            .max_by(|(a, oa), (b, ob)| {
                let sa = a.position.x + oa * 1000.0;
                let sb = b.position.x + ob * 1000.0;
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((p, _)) = best_supporter {
            // Lead the kick past the supporter toward goal (kick into space to run onto).
            let lead = (opp_goal - p.position).normalize() * config::SUPPORTER_LEAD;
            return Some(p.position + lead);
        }

        // 2. Open forward space toward goal.
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        geometry::best_pass_area(carrier_pos, opps, half_len, half_wid)
    }

    /// A small step toward the opponent goal for a corrective dribble, clamped to
    /// the field. Kept short — dribbling is a correction, not advancement.
    fn correction_target(&self, from: Vector2, opp_goal: Vector2, world: &World) -> Vector2 {
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let dir = opp_goal - from;
        let n = dir.norm();
        let step = if n > 1e-6 {
            dir / n * config::DRIBBLE_CORRECTION_STEP
        } else {
            Vector2::new(config::DRIBBLE_CORRECTION_STEP, 0.0)
        };
        let raw = from + step;
        Vector2::new(
            raw.x.clamp(-half_len + 200.0, half_len - 200.0),
            raw.y.clamp(-half_wid + 200.0, half_wid - 200.0),
        )
    }
}
