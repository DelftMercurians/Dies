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
    /// Ball is pinned against a field boundary. Approach from inside and dribble
    /// it back into play (inward heading) rather than chasing it over the line —
    /// the capturer would otherwise drive off the pitch and carry it out.
    RescueInward,
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
    /// A double-touch-safe restart release: strike the (stationary) ball forward
    /// without ever holding it, via a reflex-armed strike-through. Used only on
    /// our attacking kickoff / free kick, where holding to aim would be a
    /// double-touch. Realised by the driver via `pickup_ball_reflex`.
    Release {
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

        // Gate: a loose ball essentially on a touchline/goal line is captured as a
        // "rescue inward" — the driver dribbles it back into the field instead of
        // pushing it over the line (which concedes the ball out). Applies to both
        // loose and steal captures; the boundary takes priority over the steal.
        let on_boundary = geometry::boundary_rescue_heading(
            ball_pos,
            opp_goal,
            world.field_length() / 2.0,
            world.field_width() / 2.0,
            config::BOUNDARY_RESCUE_MARGIN,
            config::RESCUE_GOAL_BIAS,
        )
        .is_some();

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

                let waypoint = if !is_kicker && world.ball_contest().is_some() {
                    // An opponent is physically pinning the ball we nominally hold.
                    // Don't advance toward goal — that drives into the presser (the
                    // old `correction_target` deadlock). Break contact instead.
                    self.contest_escape(world, ball_pos, carrier_pos)
                } else if is_kicker {
                    // Restart: always release forward (never dribble — double-touch).
                    // Kick at a supporter if one is well placed, else open space.
                    // A `Release` is a strike-through (reflex kick), never a hold,
                    // so the kicker can't double-touch while aiming.
                    let target = self
                        .best_kickahead_target(world, id, inputs)
                        .map(|(_, t)| t)
                        .unwrap_or_else(|| self.release_target(carrier_pos, opp_goal, world));
                    Waypoint::Release { target }
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
                let kind = if on_boundary {
                    CaptureKind::RescueInward
                } else {
                    CaptureKind::Loose
                };
                Plan {
                    waypoints: vec![Waypoint::Capture { kind, robot }],
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
                let kind = if on_boundary {
                    CaptureKind::RescueInward
                } else {
                    CaptureKind::Steal { from: oid }
                };
                Plan {
                    waypoints: vec![Waypoint::Capture { kind, robot }],
                    active_robot: robot,
                }
            }
        };

        self.current_plan = Some(plan.clone());
        Some(plan)
    }

    /// Break out of a contest where an opponent is pinning the ball we hold.
    /// Threat-blended: near our own goal we clear hard to a wing (accepting the
    /// loss of possession to kill the danger and avoid a force-restart reshuffle);
    /// elsewhere we strafe perpendicular to the squeeze axis to keep the ball and
    /// move the presser out from between us and goal.
    fn contest_escape(&self, world: &World, ball_pos: Vector2, carrier_pos: Vector2) -> Waypoint {
        let own_goal = world.own_goal_center();
        let threat = geometry::threat(
            ball_pos,
            own_goal,
            config::THREAT_GOAL_NEAR,
            config::THREAT_GOAL_FAR,
        );
        if threat > config::SHADOW_RELIEF_THREAT {
            // Defensive third: firm clear toward a wing, away from our goal mouth.
            let sign = if ball_pos.y >= 0.0 { 1.0 } else { -1.0 };
            let target = Vector2::new(config::CLEAR_TARGET_X, sign * config::CLEAR_TARGET_MIN_Y);
            Waypoint::Shoot { target }
        } else {
            // Keep the ball: strafe off-axis toward the more open side. The presser
            // position falls back to the ball if the id can't be resolved (the
            // geometry helper degrades gracefully).
            let presser = world
                .principal_presser()
                .and_then(|id| world.opp_player(id))
                .map(|p| p.position)
                .unwrap_or(ball_pos);
            let dir = geometry::escape_direction(
                carrier_pos,
                presser,
                ball_pos,
                world.opp_players(),
                world.field_width() / 2.0,
            );
            Waypoint::Dribble {
                target_area: carrier_pos + dir * config::ESCAPE_STEP,
            }
        }
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
            let target = self.clamp_in_field(p.position, p.position + lead, world);
            return Some((Some(p.id), target));
        }

        // 1b. Final-third cross/cutback. Near the opponent goal there is little
        //     field left ahead, so the strict "forward" gate above finds nothing.
        //     Accept a wide, roughly-level supporter (laterally separated, not far
        //     behind) with an open lane — the classic cross/cutback outlet. Pick the
        //     one closest to the opponent goal.
        if carrier_pos.x > config::FINAL_THIRD_X {
            let best_wide = world
                .own_players()
                .iter()
                .filter(|p| p.id != carrier)
                .filter(|p| Some(p.id) != inputs.keeper_id)
                .filter(|p| Some(p.id) != inputs.double_touch_robot)
                .filter(|p| p.position.x > carrier_pos.x - config::CROSS_BACK_MARGIN)
                .filter(|p| (p.position.y - carrier_pos.y).abs() > config::CROSS_MIN_LATERAL)
                .filter(|p| {
                    geometry::lane_openness(
                        carrier_pos,
                        p.position,
                        opps,
                        config::KICK_LANE_CORRIDOR,
                    ) >= config::SUPPORTER_MIN_OPENNESS
                })
                .min_by(|a, b| {
                    let da = (opp_goal - a.position).norm();
                    let db = (opp_goal - b.position).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(p) = best_wide {
                let lead = (opp_goal - p.position).normalize() * config::SUPPORTER_LEAD;
                let target = self.clamp_in_field(p.position, p.position + lead, world);
                return Some((Some(p.id), target));
            }
        }

        // 2. Open forward space toward goal. A full-power advancement kick rolls
        //    ~HOOF_TRAVEL and would sail out of bounds if aimed naively at an
        //    open-space point a short way ahead (a stoppage + opponent free kick),
        //    so aim at the kick's resting point and keep it inside the field.
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let origin = world.ball_position().unwrap_or(carrier_pos);
        geometry::safe_kick_target(
            origin,
            opps,
            opp_goal,
            half_len,
            half_wid,
            config::HOOF_TRAVEL,
            config::HOOF_BOUNDARY_MARGIN,
            config::HOOF_MIN_PROGRESS,
            config::HOOF_OPEN_WEIGHT,
            config::HOOF_OPEN_CAP,
        )
        .map(|t| (None, t))
    }

    /// Pull a kick-ahead lead target inside the field, preserving the lead
    /// direction. The lead aims a pass into space ahead of a receiver (toward the
    /// opponent goal); for a deep or wide receiver that space can fall past the
    /// goal line or a touchline, which would send the ball out (stoppage + lost
    /// possession). We shorten the lead along its own direction so the aim point
    /// lands `FIELD_LEAD_MARGIN` inside the boundary, rather than axis-clamping it
    /// (which would skew the aim sideways). `from` is the receiver position; if it
    /// is itself outside the inset (a receiver already on the line), a hard clamp
    /// backstops the result.
    fn clamp_in_field(&self, from: Vector2, target: Vector2, world: &World) -> Vector2 {
        let max_x = world.field_length() / 2.0 - config::FIELD_LEAD_MARGIN;
        let max_y = world.field_width() / 2.0 - config::FIELD_LEAD_MARGIN;

        if target.x.abs() <= max_x && target.y.abs() <= max_y {
            return target;
        }

        // Largest fraction of the lead that keeps the endpoint inside each bound.
        let lead = target - from;
        let mut s = 1.0_f64;
        if lead.x.abs() > 1e-6 {
            let bound = if lead.x > 0.0 { max_x } else { -max_x };
            s = s.min((bound - from.x) / lead.x);
        }
        if lead.y.abs() > 1e-6 {
            let bound = if lead.y > 0.0 { max_y } else { -max_y };
            s = s.min((bound - from.y) / lead.y);
        }
        let s = s.clamp(0.0, 1.0);
        let p = from + lead * s;
        Vector2::new(p.x.clamp(-max_x, max_x), p.y.clamp(-max_y, max_y))
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
    use dies_strategy_protocol::{BallContest, BallState, GameState, WorldSnapshot};

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
            ball_contest: None,
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

    fn world_contest(
        ball: Vector2,
        own: Vec<PlayerState>,
        opp: Vec<PlayerState>,
        contest: Option<BallContest>,
    ) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball,
                velocity: Vector2::new(0.0, 0.0),
                detected: true,
            }),
            own_players: own,
            opp_players: opp,
            game_state: GameState::Run,
            us_operating: true,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::We(PlayerId::new(2)),
            possession_stale: false,
            ball_contest: contest,
        })
    }

    #[test]
    fn contest_near_our_goal_clears_to_a_wing() {
        // We hold the ball deep in our half with an opponent pressing → high threat
        // → clear hard to a wing (give up possession to kill the danger).
        let ball = Vector2::new(-3500.0, 200.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -3500.0, 200.0)];
        let opp = vec![player(9, -3300.0, 200.0)];
        let contest = Some(BallContest {
            ours: vec![PlayerId::new(2)],
            opp: vec![PlayerId::new(9)],
        });
        let world = world_contest(ball, own, opp, contest);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Shoot { target } => {
                assert!((target.x - config::CLEAR_TARGET_X).abs() < 1.0);
                assert!(target.y.abs() >= config::CLEAR_TARGET_MIN_Y - 1.0);
            }
            other => panic!("expected a clearance Shoot, got {other:?}"),
        }
    }

    #[test]
    fn contest_in_attacking_half_strafes_off_axis() {
        // We hold the ball upfield with an opponent pressing from ahead (+x) → low
        // threat → strafe laterally to keep the ball, a step of ESCAPE_STEP.
        let carrier = Vector2::new(2000.0, 300.0);
        let ball = carrier;
        let own = vec![player(1, -4000.0, 0.0), player(2, carrier.x, carrier.y)];
        let opp = vec![player(9, 2300.0, 300.0)];
        let contest = Some(BallContest {
            ours: vec![PlayerId::new(2)],
            opp: vec![PlayerId::new(9)],
        });
        let world = world_contest(ball, own, opp, contest);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Dribble { target_area } => {
                let delta = target_area - carrier;
                assert!(
                    (delta.norm() - config::ESCAPE_STEP).abs() < 1.0,
                    "should step exactly ESCAPE_STEP, got {}",
                    delta.norm()
                );
                assert!(
                    delta.x.abs() < delta.y.abs(),
                    "presser ahead (+x) → escape should be lateral (y), got {delta:?}"
                );
            }
            other => panic!("expected a strafe Dribble, got {other:?}"),
        }
    }

    #[test]
    fn attacking_restart_kicker_emits_release() {
        // On our attacking restart, the kicker holding the ball must release it
        // with a strike-through (no hold), i.e. a `Release` waypoint — never a
        // `Shoot` (DribbleShoot, which holds and would double-touch).
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -1000.0, 0.0), // kicker with the ball
        ];
        let world = world_with(own, vec![], 1);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true;

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inp)
            .expect("should produce a plan");
        assert!(
            matches!(plan.waypoints[0], Waypoint::Release { .. }),
            "expected a Release waypoint, got {:?}",
            plan.waypoints[0]
        );
        assert_eq!(plan.active_robot, PlayerId::new(2));
    }

    #[test]
    fn open_play_carrier_does_not_emit_release() {
        // Without an attacking restart, the same carrier must NOT use Release
        // (open play keeps DribbleShoot / Pass / Dribble untouched).
        let own = vec![player(1, -4000.0, 0.0), player(2, -1000.0, 0.0)];
        let world = world_with(own, vec![], 1);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");
        assert!(
            !matches!(plan.waypoints[0], Waypoint::Release { .. }),
            "open play must not emit Release, got {:?}",
            plan.waypoints[0]
        );
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

    #[test]
    fn on_ball_presser_does_not_mask_open_supporter() {
        // Regression for the "pass into the void" bug (yellow p3 ~frame 380):
        // an opponent pressed on the carrier sits at the shared origin of every
        // forward lane and, before the fix, tanked the openness of an otherwise
        // wide-open downfield supporter — so the planner never passed and hoofed
        // the ball into the corner. Positions are the real frame in team-relative
        // coordinates. The presser is ~209mm from the carrier (an on-ball contest),
        // the supporter's lane is clear apart from it.
        let own = vec![
            player(1, -4000.0, 0.0),   // keeper
            player(2, -1395.8, 539.2), // carrier (yellow p3)
            player(3, 2024.1, 1080.0), // open forward supporter (yellow p4)
        ];
        let opp = vec![player(9, -1187.0, 544.4)]; // on-ball presser (blue p3)
        let world = world_with(own, opp, 1);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");

        match &plan.waypoints[0] {
            Waypoint::Pass {
                passer, receiver, ..
            } => {
                assert_eq!(*passer, PlayerId::new(2));
                assert_eq!(
                    *receiver,
                    PlayerId::new(3),
                    "should pass to the open supporter"
                );
            }
            other => panic!("on-ball presser masked the supporter; expected a Pass, got {other:?}"),
        }
    }

    #[test]
    fn kickahead_lead_is_clamped_inside_the_field() {
        // Carrier in the final third, supporter near the opponent goal line. The
        // raw lead toward goal would push the aim point past the goal line (out of
        // bounds); the planner must clamp it inside the field while keeping the
        // pass to that receiver.
        let own = vec![
            player(1, -4000.0, 0.0),  // keeper
            player(2, 3500.0, 0.0),   // carrier in the final third
            player(3, 4200.0, 600.0), // forward supporter near the goal line
        ];
        // Opponent parked in front of goal so the direct shot is blocked (forces
        // the kick-ahead branch) but the carrier→supporter lane stays open.
        let opp = vec![player(9, 4000.0, 0.0)];
        let world = world_with(own, opp, 1);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), &inputs())
            .expect("should produce a plan");

        let max_x = world.field_length() / 2.0 - config::FIELD_LEAD_MARGIN;
        let max_y = world.field_width() / 2.0 - config::FIELD_LEAD_MARGIN;
        match &plan.waypoints[0] {
            Waypoint::Pass {
                receiver,
                target_area,
                ..
            } => {
                assert_eq!(*receiver, PlayerId::new(3));
                assert!(
                    target_area.x <= max_x + 1e-6,
                    "lead x not clamped in-field: {target_area:?}"
                );
                assert!(
                    target_area.y.abs() <= max_y + 1e-6,
                    "lead y not clamped in-field: {target_area:?}"
                );
            }
            other => panic!("expected a Pass waypoint, got {other:?}"),
        }
    }

    #[test]
    fn final_third_carrier_passes_to_wide_level_supporter() {
        // Carrier deep in the opponent third with the direct shot blocked. The only
        // outlet is a wide supporter that is NOT strictly forward (a cross/cutback).
        // The strict-forward gate rejects it; the final-third branch must accept it.
        let own = vec![
            player(1, -4000.0, 0.0),   // keeper
            player(2, 3500.0, 0.0),    // carrier in the final third
            player(3, 3600.0, 1800.0), // wide, ~level supporter (cross target)
        ];
        // Opponent parked in front of goal so the direct shot is not clear.
        let opp = vec![player(9, 4000.0, 0.0)];
        let world = world_with(own, opp, 1);
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
            other => panic!("expected a cross Pass waypoint, got {other:?}"),
        }
    }
}
