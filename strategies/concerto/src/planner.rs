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
use crate::geometry;

/// A desired ball-state transition.
///
/// Acquire + carry + shoot/strike all collapse into a single [`Waypoint::Handle`]
/// realised by the unified [`BallAction`] skill: because acquire and the terminal
/// action are one skill *variant*, a possession-change replan that swaps the
/// action is a live param update (no teardown), so the capture→act seam keeps its
/// state. Only stealing a held ball (a snatch) and a coordinated pass live outside
/// it.
#[derive(Debug, Clone)]
pub enum Waypoint {
    /// Strip a held ball off an opponent (snatch). The only capture realised
    /// outside the unified ball-handling skill — it has a distinct completion
    /// contract (succeed when the opponent loses the ball) and its own timeout.
    Steal { from: PlayerId },
    /// Acquire the ball (if not held) and perform `action` with it. `rescue`
    /// biases the acquire approach inward off a boundary. Realised each tick via
    /// `player.handle_ball`.
    Handle { action: BallAction, rescue: bool },
    /// A coordinated two-robot pass. The planner picks a concrete receiver
    /// (Model B); the framework's `PassCoordinator` drives both robots. The
    /// driver realises this by calling `ctx.pass()` each tick.
    Pass {
        passer: PlayerId,
        receiver: PlayerId,
        target_area: Vector2,
    },
}

/// Heading from `from` pointing toward the opponent goal (carry/hold facing).
fn goalward_heading(from: Vector2, opp_goal: Vector2) -> Angle {
    let d = opp_goal - from;
    Angle::from_radians(d.y.atan2(d.x))
}

/// Effective keeper shot-shadow radius as a function of how far the keeper has
/// advanced off its goal line (mm). On its line → full inflated shadow; fully
/// committed forward → plain field-robot radius; linear in between. See
/// [`config::KEEPER_CHARGE_NEAR`]/[`config::KEEPER_CHARGE_FAR`].
fn keeper_charge_radius(advance: f64) -> f64 {
    let full = config::SHOT_KEEPER_RADIUS;
    let t = ((advance - config::KEEPER_CHARGE_NEAR)
        / (config::KEEPER_CHARGE_FAR - config::KEEPER_CHARGE_NEAR))
        .clamp(0.0, 1.0);
    full + t * (config::SHOT_ROBOT_RADIUS - full)
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
    /// True for a short window right after we won the ball (see
    /// [`config::FAST_BREAK_WINDOW`]). Arms the transient fast-break commit when it
    /// coincides with a numerical break goal-side of the ball.
    pub fresh_possession: bool,
    pub now: f64,
}

/// Selects plans on discrete events. On a waypoint failure we always re-engage the
/// same most-eligible robot (never rotate a fresh robot into a contest); a fresh
/// plan is simply re-derived from the current world state.
pub struct Planner {
    current_plan: Option<Plan>,
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner {
    pub fn new() -> Self {
        Self { current_plan: None }
    }

    pub fn current_plan(&self) -> Option<&Plan> {
        self.current_plan.as_ref()
    }

    pub fn clear_plan(&mut self) {
        self.current_plan = None;
    }

    /// Re-evaluate and produce a plan, or `None` to defer to Formation (defend).
    /// `capturer` is the ball-winner chosen by Formation's matching (for a ball we
    /// don't hold); it is ignored when we already hold the ball.
    pub fn replan(
        &mut self,
        world: &World,
        possession: &Possession,
        capturer: Option<PlayerId>,
        inputs: &PlanInputs,
    ) -> Option<Plan> {
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

                // Open-goal shot: aim at the widest open window in the mouth (keeper
                // special-cased), not the centre. Gated on the window's angular
                // width (distance-aware) and on being in range. This takes priority
                // over passing — a real shooting chance is never traded for a pass.
                let shot = geometry::best_shot(
                    ball_pos,
                    opp_goal.x,
                    world.goal_width(),
                    world.opp_players(),
                    Self::opp_keeper(world),
                    config::SHOT_ROBOT_RADIUS,
                    Self::keeper_shot_radius(world),
                    config::BALL_RADIUS,
                    config::SHOT_KEEPER_BIAS,
                )
                .filter(|s| s.angle >= config::SHOT_MIN_ANGLE);
                let in_range = (carrier_pos - opp_goal).norm() < config::SHOOT_RANGE;

                // Transient fast-break: just after a turnover, in the attacking
                // half, with no opponent pinning the ball and a genuine numerical
                // break goal-side, commit the ball forward instead of recycling.
                let fast_break = inputs.fresh_possession
                    && carrier_pos.x > 0.0
                    && world.ball_contest().is_none()
                    && self.numerical_break(world, id, carrier_pos, inputs);

                let waypoint = if !is_kicker && world.ball_contest().is_some() {
                    // An opponent is physically pinning the ball we nominally hold.
                    // In the attacking strike zone, finish first-time — striking the
                    // contested ball goalward beats trying to keep it and losing the
                    // 50/50. Otherwise don't advance toward goal (that drives into the
                    // presser — the old `correction_target` deadlock): break contact.
                    self.contested_finish(world, ball_pos, opp_goal, carrier_pos)
                        .unwrap_or_else(|| self.contest_escape(world, ball_pos, carrier_pos))
                } else if is_kicker {
                    // Restart: always release forward (never dribble — double-touch).
                    // Kick at a supporter if one is well placed, else open space.
                    // A `Strike` is a strike-through (reflex kick), never a hold,
                    // so the kicker can't double-touch while aiming.
                    let target = self
                        .best_kickahead_target(world, id, inputs)
                        .map(|(_, t, _)| t)
                        .unwrap_or_else(|| self.release_target(carrier_pos, opp_goal, world));
                    Waypoint::Handle {
                        action: BallAction::Strike {
                            target,
                            acquire_first: false,
                        },
                        rescue: false,
                    }
                } else if let (true, Some(aim)) = (in_range, shot) {
                    // A real aimed shot is on. Close to goal (inside the
                    // strike-finish zone), finish it FIRST-TIME with a reflex
                    // `Strike` rather than the orbit-to-aim `Shoot`: the aimed
                    // Shoot's kick gate refuses any marginally-covered corridor and
                    // orbits indefinitely instead of firing — the dithering observed
                    // in the finishing battery, where a carrier sits on a central box
                    // position for seconds and never gets a shot away. From inside
                    // the zone aim precision matters far less than decisiveness (the
                    // window midpoint is still the target), and a strike-through fires
                    // immediately, forcing a save/rebound/goal instead of holding the
                    // ball until the defence recovers. Farther out (between
                    // STRIKE_FINISH_RANGE and SHOOT_RANGE) keep the precise aimed
                    // Shoot — there the lane is cleaner and accuracy is what pays.
                    let close = carrier_pos.x > 0.0
                        && (carrier_pos - opp_goal).norm() < config::STRIKE_FINISH_RANGE;
                    let action = if close {
                        BallAction::Strike {
                            target: aim.target,
                            acquire_first: false,
                        }
                    } else {
                        BallAction::Shoot { target: aim.target }
                    };
                    Waypoint::Handle {
                        action,
                        rescue: false,
                    }
                } else {
                    // Primary advancement is a kick-ahead toward a supporter or open
                    // space (dribbling is unreliable and rule-capped). When the best
                    // forward option is weak — a low-quality pass, or a hoof into
                    // space — and we're in the attacking half, lay the ball back to
                    // the recycle pivot instead to keep possession. *** PASS SEAM ***.
                    let forward = self.best_kickahead_target(world, id, inputs);
                    let recycle = self.recycle_target(world, id, inputs);
                    let as_pass = |(receiver, target_area): (PlayerId, Vector2)| Waypoint::Pass {
                        passer: id,
                        receiver,
                        target_area,
                    };
                    let hoof = |target: Vector2| Waypoint::Handle {
                        action: BallAction::Shoot { target },
                        rescue: false,
                    };

                    // Strike-finish fallback: in genuine shooting range with the ball
                    // but no *clean* aimed-shot window and no good forward pass, the
                    // aimed Shoot would orbit forever (its kick gate refuses a
                    // `lane_blocked` corridor and never fires from wide/covered
                    // angles), and a backward recycle from this deep repeatedly turns
                    // into a lost-territory counter (observed in the finishing
                    // battery). Instead release a one-touch Strike (reflex
                    // kick-through) at the widest available window — or, if the mouth
                    // is fully crowded, the goal centre. The Strike fires through a
                    // marginally covered lane, forcing a save/rebound/corner and
                    // pinning the ball in the attacking third. The classic coach call:
                    // in the box with nothing on, shoot — don't pass backward. It
                    // only pre-empts the *recycle* fallbacks below, never a real
                    // forward pass or cross.
                    let strike_zone = carrier_pos.x > 0.0
                        && (carrier_pos - opp_goal).norm() < config::STRIKE_FINISH_RANGE;
                    let finish = || {
                        let target = self.finish_strike_target(world, ball_pos, opp_goal);
                        Waypoint::Handle {
                            action: BallAction::Strike {
                                target,
                                acquire_first: false,
                            },
                            rescue: false,
                        }
                    };
                    // The recycle outlet, but strike instead when in shooting range.
                    let recycle_or_finish = || {
                        if strike_zone {
                            Some(finish())
                        } else {
                            recycle.map(as_pass)
                        }
                    };

                    match forward {
                        Some((Some(receiver), target_area, quality)) => {
                            // A weak forward pass is recycled when a safe outlet
                            // exists — unless we're on a fast break, where we commit
                            // the forward ball even if its static lane reads weak
                            // (the defence is still recovering, so the lane is more
                            // open than the snapshot suggests).
                            if !fast_break && quality < config::FORWARD_OK_BAR {
                                recycle_or_finish().unwrap_or(Waypoint::Pass {
                                    passer: id,
                                    receiver,
                                    target_area,
                                })
                            } else {
                                Waypoint::Pass {
                                    passer: id,
                                    receiver,
                                    target_area,
                                }
                            }
                        }
                        // About to hoof into open space: recycle instead if we can —
                        // but on a fast break, drive the ball forward into the space
                        // behind their line rather than laying it back.
                        Some((None, target, _)) => {
                            if fast_break {
                                hoof(target)
                            } else {
                                recycle_or_finish().unwrap_or_else(|| hoof(target))
                            }
                        }
                        // Nothing forward at all: strike if in range, else recycle,
                        // else a short corrective dribble (under the carry cap), else
                        // release.
                        None => {
                            if let Some(w) = recycle_or_finish() {
                                w
                            } else if inputs.carried < config::DRIBBLE_CORRECTION_LIMIT {
                                let to = self.correction_target(carrier_pos, opp_goal, world);
                                Waypoint::Handle {
                                    action: BallAction::Carry {
                                        to,
                                        heading: goalward_heading(to, opp_goal),
                                    },
                                    rescue: false,
                                }
                            } else {
                                hoof(self.release_target(carrier_pos, opp_goal, world))
                            }
                        }
                    }
                };
                Plan {
                    waypoints: vec![waypoint],
                    active_robot: id,
                }
            }

            // ── Ball is loose (or contested — go win it) ────────────────
            Possession::Loose | Possession::Contested => {
                let robot = capturer?;
                // First-time finish: a contested loose ball in the attacking strike
                // zone (and not a boundary rescue) is struck goalward on contact — a
                // genuine 50/50 in the final third is lost when we try to settle it,
                // and the recycle/loss becomes a counter (finishing battery). Only
                // when our capturer is right on it (a reflex strike is imminent).
                let actor_pos = world
                    .own_player(robot)
                    .map(|p| p.position)
                    .unwrap_or(ball_pos);
                let finish = (!on_boundary)
                    .then(|| self.contested_finish(world, ball_pos, opp_goal, actor_pos))
                    .flatten();
                // Acquire and hold; the next replan (on the possession flip) swaps
                // this Hold to the real action as a live param update — same skill
                // instance, no teardown. `rescue` biases the approach inward off a
                // boundary; the driver computes the live approach heading.
                let waypoint = finish.unwrap_or(Waypoint::Handle {
                    action: BallAction::Hold {
                        heading: goalward_heading(ball_pos, opp_goal),
                    },
                    rescue: on_boundary,
                });
                Plan {
                    waypoints: vec![waypoint],
                    active_robot: robot,
                }
            }

            // ── Opponent has the ball ───────────────────────────────────
            Possession::Opp(oid) => {
                // Formation already elected this challenger through the capture
                // time-horizon gate. Accept it as-is: re-gating here (e.g. on
                // distance) could strip a robot of its defensive role yet leave it
                // uncommanded, since Formation has excluded it from positioning.
                let robot = capturer?;
                // A ball pinned on a boundary is rescued inward (acquire + hold),
                // taking priority over the steal; otherwise strip it from the holder.
                let waypoint = if on_boundary {
                    Waypoint::Handle {
                        action: BallAction::Hold {
                            heading: goalward_heading(ball_pos, opp_goal),
                        },
                        rescue: true,
                    }
                } else {
                    Waypoint::Steal { from: oid }
                };
                Plan {
                    waypoints: vec![waypoint],
                    active_robot: robot,
                }
            }
        };

        self.current_plan = Some(plan.clone());
        Some(plan)
    }

    /// A genuine numerical break: strictly more of our own field robots are
    /// goal-side of the ball (carrier + keeper excluded) than the opponent has
    /// (its keeper excluded), and at least one of ours is up there to receive.
    /// This is the textbook counter-attack situation — we out-man the recovering
    /// defence, so the direct forward ball is worth more than keeping possession.
    fn numerical_break(
        &self,
        world: &World,
        carrier: PlayerId,
        carrier_pos: Vector2,
        inputs: &PlanInputs,
    ) -> bool {
        let own_ahead = world
            .own_players()
            .iter()
            .filter(|p| p.id != carrier)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| p.position.x > carrier_pos.x)
            .count();
        let opp_keeper = Self::opp_keeper(world);
        let opp_ahead = world
            .opp_players()
            .iter()
            .filter(|p| Some(p.id) != opp_keeper)
            .filter(|p| p.position.x > carrier_pos.x)
            .count();
        own_ahead >= 1 && own_ahead > opp_ahead
    }

    /// Infer the opponent keeper: the opponent nearest its own goal (the goal we
    /// attack). Strategies aren't told the opponent's designated keeper, but the
    /// robot parked deepest in front of the mouth is it in practice — and if no
    /// opponent is near the goal, the nearest one casts no central shadow anyway, so
    /// the shot model degrades gracefully.
    fn opp_keeper(world: &World) -> Option<PlayerId> {
        let goal = world.opp_goal_center();
        world
            .opp_players()
            .iter()
            .min_by(|a, b| {
                let da = (a.position - goal).norm();
                let db = (b.position - goal).norm();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id)
    }

    /// Effective shadow radius for the opponent keeper in the shot model, shrunk
    /// when the keeper has committed forward off its goal line. On its line it can
    /// dive/slide to cover the mouth, so it earns the full inflated
    /// [`config::SHOT_KEEPER_RADIUS`]; charged up-field it cannot recover to the
    /// corners, so its shadow collapses toward a plain field robot's. Linearly
    /// interpolated by how far the keeper sits off its line (along x, toward us).
    fn keeper_shot_radius(world: &World) -> f64 {
        let full = config::SHOT_KEEPER_RADIUS;
        let goal = world.opp_goal_center();
        let Some(kid) = Self::opp_keeper(world) else {
            return full;
        };
        let Some(keeper) = world.opp_players().iter().find(|p| p.id == kid) else {
            return full;
        };
        // Advancement off the goal line: the goal we attack is at max +x, so a
        // keeper that has charged toward the ball has a *smaller* x than its goal.
        let advance = (goal.x - keeper.position.x).max(0.0);
        keeper_charge_radius(advance)
    }

    /// Break out of a contest where an opponent is pinning the ball we hold.
    /// Threat-blended: near our own goal we clear hard to a wing (accepting the
    /// loss of possession to kill the danger and avoid a force-restart reshuffle);
    /// elsewhere we strafe perpendicular to the squeeze axis to keep the ball and
    /// move the presser out from between us and goal.
    /// Aim point for a one-touch finish: the widest open window in the goal mouth
    /// (`best_shot`), or the goal centre when the mouth is fully crowded.
    fn finish_strike_target(&self, world: &World, ball_pos: Vector2, opp_goal: Vector2) -> Vector2 {
        geometry::best_shot(
            ball_pos,
            opp_goal.x,
            world.goal_width(),
            world.opp_players(),
            Self::opp_keeper(world),
            config::SHOT_ROBOT_RADIUS,
            Self::keeper_shot_radius(world),
            config::BALL_RADIUS,
            config::SHOT_KEEPER_BIAS,
        )
        .map(|s| s.target)
        .unwrap_or(opp_goal)
    }

    /// First-time finish: when a *contested* ball sits in the attacking strike zone
    /// and a reflex strike is imminent (both an opponent and our actor are right on
    /// it), striking it goalward on contact beats trying to settle the 50/50 — which
    /// loses the ball and concedes a counter (the finishing-battery failure). Returns
    /// the reflex-`Strike` waypoint when the finish is on, else `None` so the caller
    /// keeps its normal behaviour (settle / escape).
    fn contested_finish(
        &self,
        world: &World,
        ball_pos: Vector2,
        opp_goal: Vector2,
        actor_pos: Vector2,
    ) -> Option<Waypoint> {
        let in_zone =
            ball_pos.x > 0.0 && (ball_pos - opp_goal).norm() < config::STRIKE_FINISH_RANGE;
        if !in_zone {
            return None;
        }
        let opp_d = world
            .closest_opp_player_to(ball_pos)
            .map(|p| (p.position - ball_pos).norm())?;
        if opp_d > config::FINISH_CONTEST_OPP_DIST
            || (actor_pos - ball_pos).norm() > config::FINISH_CONTEST_OUR_DIST
        {
            return None;
        }
        let target = self.finish_strike_target(world, ball_pos, opp_goal);
        Some(Waypoint::Handle {
            action: BallAction::Strike {
                target,
                acquire_first: false,
            },
            rescue: false,
        })
    }

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
            Waypoint::Handle {
                action: BallAction::Shoot { target },
                rescue: false,
            }
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
            let to = carrier_pos + dir * config::ESCAPE_STEP;
            Waypoint::Handle {
                action: BallAction::Carry {
                    to,
                    heading: goalward_heading(to, world.opp_goal_center()),
                },
                rescue: false,
            }
        }
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

    /// Best kick-ahead target. Returns `(Some(supporter), lead_point, quality)` for
    /// a well-placed forward supporter (the offense path promotes this to a Pass),
    /// `(None, open_point, 0.0)` for open forward space (a plain Shoot), or `None`
    /// when congested. `quality` is the pass-success of the chosen outlet (0 for a
    /// hoof), so the caller can recycle over a weak forward option.
    fn best_kickahead_target(
        &self,
        world: &World,
        carrier: PlayerId,
        inputs: &PlanInputs,
    ) -> Option<(Option<PlayerId>, Vector2, f64)> {
        let carrier_pos = world.own_player(carrier)?.position;
        let opp_goal = world.opp_goal_center();
        let opps = world.opp_players();
        // Pass-success quality of an outlet: lane openness × clearance of the receive
        // point (a marker sitting on the receiver intercepts on arrival even down an
        // open lane), scaled by the global pass-propensity base. Returned so the
        // caller can weigh a forward pass vs a recycle. At `base == 0` every outlet
        // reads as unusable, so no forward pass (or cross) is ever eligible.
        let pass_quality = |to: Vector2| -> f64 {
            config::PASS_SUCCESS_BASE
                * geometry::lane_openness(carrier_pos, to, opps, config::KICK_LANE_CORRIDOR)
                * geometry::receiver_clearance(to, opps)
        };
        let pass_ok = |to: Vector2| -> bool {
            config::PASS_SUCCESS_BASE
                * geometry::lane_openness(carrier_pos, to, opps, config::KICK_LANE_CORRIDOR)
                >= config::SUPPORTER_MIN_OPENNESS
        };

        // 1. Forward supporter with an open lane from the ball. Score by how far
        //    forward (goalward) it is, weighted by lane openness.
        let outlet_score = |p: &PlayerState| -> f64 {
            let open =
                geometry::lane_openness(carrier_pos, p.position, opps, config::KICK_LANE_CORRIDOR);
            p.position.x + open * 1000.0
        };
        let best_supporter = world
            .own_players()
            .iter()
            .filter(|p| p.id != carrier)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| Some(p.id) != inputs.double_touch_robot)
            .filter(|p| p.position.x > carrier_pos.x + config::SUPPORTER_FWD_MARGIN)
            .filter(|p| pass_ok(p.position))
            .max_by(|a, b| {
                outlet_score(a)
                    .partial_cmp(&outlet_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(p) = best_supporter {
            // Lead the kick past the supporter toward goal (kick into space to run onto).
            let lead = (opp_goal - p.position).normalize() * config::SUPPORTER_LEAD;
            let target = self.clamp_in_field(p.position, p.position + lead, world);
            return Some((Some(p.id), target, pass_quality(p.position)));
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
                .filter(|p| pass_ok(p.position))
                .min_by(|a, b| {
                    let da = (opp_goal - a.position).norm();
                    let db = (opp_goal - b.position).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(p) = best_wide {
                let lead = (opp_goal - p.position).normalize() * config::SUPPORTER_LEAD;
                let target = self.clamp_in_field(p.position, p.position + lead, world);
                return Some((Some(p.id), target, pass_quality(p.position)));
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
        // A hoof into open space keeps no possession (quality 0): the eager-recycle
        // path treats it as a last resort, preferring a safe recycle when one exists.
        .map(|t| (None, t, 0.0))
    }

    /// A backward/lateral *recycle* outlet — the most open teammate behind
    /// or level with the carrier, far enough to be a real switch of play. Lets a
    /// carrier with nothing forward keep possession (lay it back to the pivot)
    /// instead of dribbling into pressure or hoofing the ball away. Returns the
    /// receiver and the aim point (the teammate itself; a recycle is not led into
    /// space, it is played to feet).
    fn recycle_target(
        &self,
        world: &World,
        carrier: PlayerId,
        inputs: &PlanInputs,
    ) -> Option<(PlayerId, Vector2)> {
        let carrier_pos = world.own_player(carrier)?.position;
        // Never circulate in our own half — a recycle/turnover there becomes a goal.
        // This attacking-half gate is what makes recycling a net win (recycling
        // everywhere conceded significantly more and lost in self-play).
        if carrier_pos.x <= 0.0 {
            return None;
        }
        let opps = world.opp_players();
        world
            .own_players()
            .iter()
            .filter(|p| p.id != carrier)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| Some(p.id) != inputs.double_touch_robot)
            // Behind or level with the carrier (a recycle, not a forward pass)...
            .filter(|p| p.position.x < carrier_pos.x + config::SUPPORTER_FWD_MARGIN)
            // ...but still in the attacking half, so we don't drag play homeward.
            .filter(|p| p.position.x > 0.0)
            .filter(|p| (p.position - carrier_pos).norm() > config::RECYCLE_MIN_DIST)
            .filter(|p| {
                // Scaled by the global pass-propensity base so the recycle outlet —
                // itself a pass — is suppressed alongside forward passes, and
                // vanishes entirely at `base == 0`.
                let open = config::PASS_SUCCESS_BASE
                    * geometry::lane_openness(
                        carrier_pos,
                        p.position,
                        opps,
                        config::KICK_LANE_CORRIDOR,
                    );
                let clear = geometry::receiver_clearance(p.position, opps);
                open >= config::RECYCLE_MIN_OPENNESS && clear >= 0.5
            })
            // Prefer the safest outlet: the most open lane plus the most clear
            // receive point (a lay-back we're confident actually completes).
            .max_by(|a, b| {
                let score = |p: &PlayerState| {
                    geometry::lane_openness(
                        carrier_pos,
                        p.position,
                        opps,
                        config::KICK_LANE_CORRIDOR,
                    ) + geometry::receiver_clearance(p.position, opps)
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| (p.id, p.position))
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
            pre_stage: false,
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
            fresh_possession: false,
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
            pre_stage: false,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::We(PlayerId::new(2)),
            possession_stale: false,
            ball_contest: contest,
        })
    }

    #[test]
    fn keeper_charge_radius_shrinks_with_advancement() {
        // On its line (no advancement) → full inflated shadow.
        assert!((keeper_charge_radius(0.0) - config::SHOT_KEEPER_RADIUS).abs() < 1e-6);
        assert!(
            (keeper_charge_radius(config::KEEPER_CHARGE_NEAR) - config::SHOT_KEEPER_RADIUS).abs()
                < 1e-6
        );
        // Fully committed forward → shrinks to a plain field-robot shadow.
        assert!(
            (keeper_charge_radius(config::KEEPER_CHARGE_FAR) - config::SHOT_ROBOT_RADIUS).abs()
                < 1e-6
        );
        assert!(
            (keeper_charge_radius(config::KEEPER_CHARGE_FAR + 5000.0) - config::SHOT_ROBOT_RADIUS)
                .abs()
                < 1e-6
        );
        // Halfway between near and far → midpoint radius, and monotone shrinking.
        let mid =
            keeper_charge_radius((config::KEEPER_CHARGE_NEAR + config::KEEPER_CHARGE_FAR) / 2.0);
        let expected = (config::SHOT_KEEPER_RADIUS + config::SHOT_ROBOT_RADIUS) / 2.0;
        assert!((mid - expected).abs() < 1e-6);
        assert!(mid < config::SHOT_KEEPER_RADIUS && mid > config::SHOT_ROBOT_RADIUS);
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
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Handle {
                action: BallAction::Shoot { target },
                ..
            } => {
                assert!((target.x - config::CLEAR_TARGET_X).abs() < 1.0);
                assert!(target.y.abs() >= config::CLEAR_TARGET_MIN_Y - 1.0);
            }
            other => panic!("expected a clearance Shoot, got {other:?}"),
        }
    }

    #[test]
    fn contest_in_attacking_half_strafes_off_axis() {
        // We hold the ball upfield (but OUTSIDE the strike-finish zone) with an
        // opponent pressing from ahead (+x) → low threat, no finish on → strafe
        // laterally to keep the ball, a step of ESCAPE_STEP.
        let carrier = Vector2::new(500.0, 300.0);
        let ball = carrier;
        let own = vec![player(1, -4000.0, 0.0), player(2, carrier.x, carrier.y)];
        let opp = vec![player(9, 800.0, 300.0)];
        let contest = Some(BallContest {
            ours: vec![PlayerId::new(2)],
            opp: vec![PlayerId::new(9)],
        });
        let world = world_contest(ball, own, opp, contest);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Handle {
                action: BallAction::Carry { to, .. },
                ..
            } => {
                let delta = to - carrier;
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
            other => panic!("expected a strafe Carry, got {other:?}"),
        }
    }

    #[test]
    fn contested_ball_in_strike_zone_finishes_first_time() {
        // Same press, but in the attacking strike zone (within STRIKE_FINISH_RANGE of
        // the opp goal): rather than strafe to keep a 50/50 we'd likely lose, strike
        // the contested ball goalward on contact (the first-time finish).
        let carrier = Vector2::new(2600.0, 200.0);
        let ball = carrier;
        let own = vec![player(1, -4000.0, 0.0), player(2, carrier.x, carrier.y)];
        let opp = vec![player(9, 2850.0, 200.0)];
        let contest = Some(BallContest {
            ours: vec![PlayerId::new(2)],
            opp: vec![PlayerId::new(9)],
        });
        let world = world_contest(ball, own, opp, contest);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            matches!(
                &plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "a contested ball in the strike zone should finish first-time, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn close_range_aimed_shot_is_a_first_time_strike() {
        // An uncontested carrier with a clean aimed shot from INSIDE the
        // strike-finish zone finishes first-time with a reflex Strike rather than
        // orbiting to aim with Shoot (which dithers when the lane is marginally
        // covered and never fires — the finishing-battery failure).
        let carrier = Vector2::new(2700.0, 0.0); // ~1800mm from goal, in zone
        let own = vec![player(1, -4000.0, 0.0), player(2, carrier.x, carrier.y)];
        let opp = vec![player(9, -4000.0, 0.0)]; // far behind the ball → open mouth
        let world = world_contest(carrier, own, opp, None);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            matches!(
                &plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "a close-range aimed shot should be a first-time Strike, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn long_range_aimed_shot_keeps_the_precise_shoot() {
        // Same open mouth but from OUTSIDE the strike-finish zone (still in range):
        // accuracy matters more than decisiveness, so keep the orbit-to-aim Shoot.
        let carrier = Vector2::new(700.0, 0.0); // ~3800mm from goal: >zone, <range
        let own = vec![player(1, -4000.0, 0.0), player(2, carrier.x, carrier.y)];
        let opp = vec![player(9, -4000.0, 0.0)];
        let world = world_contest(carrier, own, opp, None);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            matches!(
                &plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Shoot { .. },
                    ..
                }
            ),
            "a long-range aimed shot should keep the precise Shoot, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn uncontested_loose_ball_in_zone_is_settled_not_struck() {
        // A loose ball in the strike zone with NO opponent near it: we win it clean,
        // so settle (Hold) and build — don't hack at goal.
        let ball = Vector2::new(2600.0, 200.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, 2400.0, 200.0)];
        let opp = vec![player(9, 3800.0, 0.0)]; // far from the ball
        let world = world_contest(ball, own, opp, None);
        let mut planner = Planner::new();

        let plan = planner
            .replan(
                &world,
                &Possession::Loose,
                Some(PlayerId::new(2)),
                &inputs(),
            )
            .expect("should produce a plan");
        assert!(
            matches!(
                &plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Hold { .. },
                    ..
                }
            ),
            "an uncontested loose ball should be settled, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn attacking_restart_kicker_emits_release() {
        // On our attacking restart, the kicker holding the ball must release it
        // with a strike-through (no hold), i.e. a `Release` waypoint — never a
        // `Shoot` (which captures and holds the ball, and would double-touch).
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -1000.0, 0.0), // kicker with the ball
        ];
        let world = world_with(own, vec![], 1);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true;

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inp)
            .expect("should produce a plan");
        assert!(
            matches!(
                plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "expected a Strike waypoint, got {:?}",
            plan.waypoints[0]
        );
        assert_eq!(plan.active_robot, PlayerId::new(2));
    }

    #[test]
    fn open_play_carrier_does_not_emit_release() {
        // Without an attacking restart, the same carrier must NOT strike-through
        // (open play keeps capture-then-aim Shoot / Pass / Carry untouched).
        let own = vec![player(1, -4000.0, 0.0), player(2, -1000.0, 0.0)];
        let world = world_with(own, vec![], 1);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            !matches!(
                plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "open play must not emit Strike, got {:?}",
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
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
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
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
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
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
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
    fn fast_break_commits_forward_instead_of_recycling() {
        // Carrier just past midfield (no shot in range), the only forward teammate
        // has a blocked lane, and a safe recycle outlet sits behind. Normally we lay
        // the ball back (keep possession). On a fast break with a numerical edge
        // goal-side, we commit the ball forward (hoof into space) instead.
        let own = vec![
            player(1, -4000.0, 0.0),  // keeper
            player(2, 400.0, 0.0),    // carrier (out of shot range)
            player(3, 1500.0, 0.0),   // forward, but its lane is blocked
            player(4, 200.0, 1300.0), // recycle outlet behind/level
        ];
        // One opponent on the carrier→forward lane; it is the opp keeper (nearest the
        // opp goal), so it is excluded from the numerical-break count → we out-man
        // the defence goal-side (1 vs 0).
        let opp = vec![player(9, 1000.0, 0.0)];

        // Without a fresh turnover: recycle (lay the ball back to player 4).
        let world = world_with(own.clone(), opp.clone(), 1);
        let mut planner = Planner::new();
        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Pass { receiver, .. } => assert_eq!(*receiver, PlayerId::new(4)),
            other => panic!("patient play should recycle to p4, got {other:?}"),
        }

        // Same frame, but fresh possession → fast break commits the ball forward.
        let world = world_with(own, opp, 1);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.fresh_possession = true;
        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inp)
            .expect("should produce a plan");
        assert!(
            matches!(
                plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Shoot { .. },
                    ..
                }
            ),
            "fast break should drive the ball forward, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn fast_break_suppressed_without_numerical_edge() {
        // Same fresh turnover, but the opponent has a field robot back goal-side
        // (in addition to its keeper), so we do NOT out-man the defence — the break
        // gate stays shut and we play patiently (recycle).
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 400.0, 0.0),
            player(3, 1500.0, 0.0),
            player(4, 200.0, 1300.0),
        ];
        // p9 is the opp keeper (deepest); p8 is a field defender goal-side of the
        // ball → opp_ahead = 1, own_ahead = 1, so own_ahead > opp_ahead is false.
        let opp = vec![player(9, 4200.0, 0.0), player(8, 1000.0, 0.0)];
        let world = world_with(own, opp, 1);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.fresh_possession = true;
        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inp)
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Pass { receiver, .. } => assert_eq!(*receiver, PlayerId::new(4)),
            other => panic!("no numerical edge → should recycle to p4, got {other:?}"),
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
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
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

    #[test]
    fn box_carrier_with_no_clean_shot_strikes() {
        // Carrier on the ball in the box, the goal mouth crowded so no *clean*
        // aimed-shot window survives the angle gate, and no forward outlet exists.
        // Rather than recycle the ball backward, the carrier must release a
        // one-touch Strike at the goal (the reflex finish that fires through a
        // covered lane).
        let ball = Vector2::new(3700.0, 0.0);
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, 3700.0, 0.0),  // carrier on the ball in the box
        ];
        // Three opponents fanned across the mouth: every open gap is too narrow to
        // clear the angle gate, so the aimed Shoot is off the table.
        let opp = vec![
            player(7, 4200.0, -300.0),
            player(8, 4200.0, 0.0),
            player(9, 4200.0, 300.0),
        ];
        let world = world_contest(ball, own, opp, None);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            matches!(
                plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "expected a Strike finish in the box, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn deep_carrier_with_no_clean_shot_does_not_strike() {
        // The same crowded-mouth situation but the carrier is in its own half,
        // far out of shooting range: the strike-finish must NOT fire there (a give
        // away that deep is a counter), it falls back to the normal recycle/carry.
        let ball = Vector2::new(-1500.0, 0.0);
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -1500.0, 0.0), // carrier deep in our own half
        ];
        let opp = vec![
            player(7, 4200.0, -300.0),
            player(8, 4200.0, 0.0),
            player(9, 4200.0, 300.0),
        ];
        let world = world_contest(ball, own, opp, None);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            !matches!(
                plan.waypoints[0],
                Waypoint::Handle {
                    action: BallAction::Strike { .. },
                    ..
                }
            ),
            "must not strike-finish from deep in our own half, got {:?}",
            plan.waypoints[0]
        );
    }
}
