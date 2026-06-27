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
    /// A coordinated two-robot pass. The planner picks a concrete receiver
    /// (Model B); the framework's `PassCoordinator` drives both robots. The
    /// driver realises this by calling `ctx.pass()` each tick.
    Pass {
        passer: PlayerId,
        receiver: PlayerId,
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

/// What the plan-controlled robots are covering this tick, exported to Formation
/// so it can account for that coverage instead of piling a second robot onto the
/// same spot (the clustering fix). This is the enriched plan→formation channel
/// that replaces the old `plan_ctx: Option<Vector2>`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanContext {
    pub engagements: Vec<Engagement>,
}

/// A single coverage fact about a plan-controlled robot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Engagement {
    /// A plan robot is contesting / holding the ball here (capture, carry, or the
    /// passer side of a pass). Formation soft-suppresses roles near this point.
    BallContest { at: Vector2 },
    /// A plan robot is pressuring this opponent (steal) — it *is* the mark, so
    /// Formation skips generating a mark role for this opponent.
    Engaging { opponent: PlayerId },
    /// The pass receive area. The receiver itself is a plan slot (Model B); this
    /// is a hint for Formation to backfill the vacated support (currently unused).
    #[allow(dead_code)]
    ReceiveArea { at: Vector2 },
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
                    // (Stays a Shoot, not a Pass: restart legality / double-touch is
                    // its own problem — see plan, out of scope.)
                    let target = self
                        .best_kickahead_target(world, id, inputs)
                        .map(|(_, t)| t)
                        .unwrap_or_else(|| self.release_target(carrier_pos, opp_goal, world));
                    Waypoint::Shoot { target }
                } else if clear && in_range {
                    Waypoint::Shoot { target: opp_goal }
                } else {
                    // Primary advancement is a kick-ahead toward a supporter or open
                    // space (dribbling is unreliable and rule-capped). Dribble only as
                    // a small correction, and only while we're under the carry cap.
                    // *** PASS SEAM ***: a well-placed supporter becomes a coordinated
                    // Pass; open space stays a Shoot (no specific receiver to commit).
                    match self.best_kickahead_target(world, id, inputs) {
                        Some((Some(receiver), target_area)) => Waypoint::Pass {
                            passer: id,
                            receiver,
                            target_area,
                        },
                        Some((None, target)) => Waypoint::Shoot { target },
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

            // ── Ball is loose (or contested — go win it) ────────────────
            Possession::Loose | Possession::Contested => {
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

    /// Best kick-ahead target. Returns `(Some(supporter), lead_point)` for a
    /// well-placed forward supporter (the offense path promotes this to a Pass),
    /// `(None, open_point)` for open forward space (a plain Shoot), or `None` when
    /// congested (no supporter and no open area) — the rare case where a small
    /// corrective dribble is warranted.
    fn best_kickahead_target(
        &self,
        world: &World,
        carrier: PlayerId,
        inputs: &PlanInputs,
    ) -> Option<(Option<PlayerId>, Vector2)> {
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
            return Some((Some(p.id), p.position + lead));
        }

        // 2. Open forward space toward goal.
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        geometry::best_pass_area(carrier_pos, opps, half_len, half_wid).map(|t| (None, t))
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

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{BallState, GameState, WorldSnapshot};

    fn player(id: u32, x: f64, y: f64) -> PlayerState {
        PlayerState::new(
            PlayerId::new(id),
            Vector2::new(x, y),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        )
    }

    fn world_with(own: Vec<PlayerState>, opp: Vec<PlayerState>, keeper: u32) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: Vector2::new(-1000.0, 0.0),
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: own,
            opp_players: opp,
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(keeper)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
        })
    }

    fn inputs() -> PlanInputs {
        PlanInputs {
            keeper_id: Some(PlayerId::new(1)),
            double_touch_robot: None,
            our_attacking_restart: false,
            carried: 0.0,
            now: 0.0,
        }
    }

    #[test]
    fn carrier_with_open_supporter_emits_pass() {
        // Carrier deep in our half (no shot in range), a clear forward supporter,
        // no opponents in the lane → the planner should commit a coordinated pass.
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -1000.0, 0.0), // carrier
            player(3, 1500.0, 0.0),  // open forward supporter
        ];
        let world = world_with(own, vec![], 1);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");

        match &plan.waypoints[0] {
            Waypoint::Pass {
                passer, receiver, ..
            } => {
                assert_eq!(*passer, PlayerId::new(2));
                assert_eq!(*receiver, PlayerId::new(3));
            }
            other => panic!("expected a Pass waypoint, got {other:?}"),
        }
        assert_eq!(plan.active_robot, PlayerId::new(2));
    }
}
