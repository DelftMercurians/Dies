//! Planner — decides where the ball should go. Nothing else.
//!
//! One decision, the **destination ladder**: a clear aimed shot, the corner
//! one-timer, a forward pass, or an advance strike down the cleanest lane. The
//! executor's `HandleBall` skill owns every mechanic below that (stage behind
//! the ball, drive-through, reflex kick) — the planner never manages acquire /
//! aim / hold / disengage, and possession (`We`/`Loose`/`Contested`) is not a
//! planning seam: all three funnel into the same decision, the actor being the
//! carrier or Formation's elected capturer. It runs only on discrete events
//! (see `lib.rs`), so plan continuity falls out for free.
//!
//! The one non-kick behaviour: a ball in **stable opponent possession** (the
//! framework's dribbler-point + Schmitt tracker) is harassed — our robot parks
//! directly in front of the holder's dribbler and blocks its advancement —
//! instead of chasing a ball we can't reach sideways.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use crate::config;
use crate::geometry;

/// A desired ball destination (or the harass block).
#[derive(Debug, Clone, PartialEq)]
pub enum Waypoint {
    /// One strike-through kick episode toward `target`: stage behind the ball,
    /// drive through, reflex-kick. Realised as
    /// `BallAction::Strike { acquire_first: false }` — the ball is never held,
    /// aimed with, or disengaged; a whiff fails the waypoint and triggers a
    /// replan. Used for every kick: shots, finishes, hoofs, clears, restarts.
    Strike { target: Vector2 },
    /// A coordinated two-robot pass. The planner picks the concrete receiver;
    /// the framework's `PassCoordinator` drives both robots.
    Pass {
        passer: PlayerId,
        receiver: PlayerId,
        target_area: Vector2,
        /// `Some(aim)` turns the receive into a one-timer: the receiver waits
        /// with the reflex kick pre-armed facing `aim` and redirects the ball
        /// there on contact. `None` is a plain catch pass.
        forward_to: Option<Vector2>,
    },
    /// Block an opponent holding the ball: stand directly in front of its
    /// dribbler so it cannot advance. Never self-completes — it ends when the
    /// possession edge triggers the next replan (holder loses the ball).
    Harass { opponent: PlayerId },
}

/// A plan: one waypoint driven by one active robot (replan-after-each makes any
/// tail advisory, so v1 emits length-1).
#[derive(Debug, Clone)]
pub struct Plan {
    pub waypoints: Vec<Waypoint>,
    pub active_robot: PlayerId,
}

/// What the plan-controlled robots are covering this tick, exported to Formation
/// so it can account for that coverage instead of piling a second robot onto the
/// same spot (the clustering fix).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PlanContext {
    pub engagements: Vec<Engagement>,
}

/// A single coverage fact about a plan-controlled robot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Engagement {
    /// A plan robot is contesting / working the ball here. Formation
    /// soft-suppresses roles near this point.
    BallContest { at: Vector2 },
    /// A plan robot is pressuring this opponent (harass) — it *is* the mark, so
    /// Formation skips generating a mark role for this opponent.
    Engaging { opponent: PlayerId },
    /// The pass receive area (a hint for Formation to backfill the vacated
    /// support; currently unused).
    #[allow(dead_code)]
    ReceiveArea { at: Vector2 },
}

/// Inputs the planner needs beyond the world snapshot.
pub struct PlanInputs {
    pub keeper_id: Option<PlayerId>,
    pub double_touch_robot: Option<PlayerId>,
    /// True during our in-play kickoff/free kick: the taker's first touch IS the
    /// kick (a strike-through from the loose ball), with the restart shot range
    /// and — for a kickoff — the deflection ladder.
    pub our_attacking_restart: bool,
    /// True when that attacking restart is specifically a KICKOFF (vs a free
    /// kick). A kickoff with no clear shot deflects off a distant side opponent.
    pub our_restart_is_kickoff: bool,
}

/// Selects plans on discrete events. On a waypoint failure we always re-engage
/// the same most-eligible robot (never rotate a fresh robot into a contest); a
/// fresh plan is simply re-derived from the current world state.
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
    /// `capturer` is the ball-winner chosen by Formation's matching (for a ball
    /// we don't hold); it is ignored when we already hold the ball.
    pub fn replan(
        &mut self,
        world: &World,
        possession: &Possession,
        capturer: Option<PlayerId>,
        inputs: &PlanInputs,
    ) -> Option<Plan> {
        let ball = world.ball_position()?;

        // The keeper is never a plan's active robot: its Guard/Clear machine owns
        // it exclusively (lib.rs also gates offense/pursuit off when the keeper
        // holds the ball — this is the backstop for every other replan path).
        if matches!(*possession, Possession::We(id) if Some(id) == inputs.keeper_id) {
            return None;
        }

        let (active_robot, waypoint) = match *possession {
            // We hold the ball, or it's loose/contested: same decision — where
            // should the ball go. The actor is the carrier or the elected capturer.
            Possession::We(id) => (id, self.destination(world, id, ball, inputs)),
            Possession::Loose | Possession::Contested => {
                let robot = capturer?;
                (robot, self.destination(world, robot, ball, inputs))
            }
            // Stable opponent possession: harass — park in front of the holder's
            // dribbler and block its advancement. EXCEPT in the wall's apron,
            // where a body parked on the holder would screen our own wall/keeper:
            // there the challenger pursues the ball itself (a plain strike).
            Possession::Opp(oid) => {
                let robot = capturer?;
                let wp = if crate::wall_owns_ball(world) {
                    self.destination(world, robot, ball, inputs)
                } else {
                    Waypoint::Harass { opponent: oid }
                };
                (robot, wp)
            }
        };

        let plan = Plan {
            waypoints: vec![waypoint],
            active_robot,
        };
        self.current_plan = Some(plan.clone());
        Some(plan)
    }

    /// The destination ladder — the planner's single decision. In priority order:
    ///   1. a clear aimed shot (the widest open goal-mouth window, angle-gated);
    ///   2. the corner one-timer (deep and wide of the box, cross for a redirect);
    ///   3. a forward pass to a well-placed supporter (never backward);
    ///   4. an advance strike: kickoff deflection / strike-finish / hoof, every
    ///      release guarded by [`Self::safe_release`].
    /// All kick decisions are evaluated from the BALL (the kick origin), so a
    /// mid-episode possession flip re-derives an equivalent `Strike` — a no-op
    /// live param swap on the running skill, never a teardown.
    fn destination(
        &self,
        world: &World,
        actor: PlayerId,
        ball: Vector2,
        inputs: &PlanInputs,
    ) -> Waypoint {
        let shot_range = if inputs.our_attacking_restart {
            config::KICKER_SHOT_RANGE
        } else {
            config::SHOOT_RANGE
        };
        if let Some(aim) = self.clear_shot(world, ball, shot_range) {
            return Waypoint::Strike { target: aim };
        }
        if let Some(pass) = self.corner_one_timer(world, actor, ball, inputs) {
            return pass;
        }
        if let Some((receiver, target_area)) = self.forward_pass(world, actor, ball, inputs) {
            return Waypoint::Pass {
                passer: actor,
                receiver,
                target_area,
                forward_to: None,
            };
        }
        Waypoint::Strike {
            target: self.advance_target(world, actor, ball, inputs),
        }
    }

    /// A direct shot at the opponent goal — taken only when a genuinely clear
    /// window is open within `range` AND the lane isn't fired into a close
    /// opponent. Open play and restarts are held to the same "actually clear"
    /// gate ([`config::SHOT_MIN_ANGLE`]).
    fn clear_shot(&self, world: &World, ball: Vector2, range: f64) -> Option<Vector2> {
        if (ball - world.opp_goal_center()).norm() >= range {
            return None;
        }
        geometry::best_shot_from(world, ball)
            .filter(|s| s.angle >= config::SHOT_MIN_ANGLE)
            .filter(|s| !self.fires_into_close_opp(world, ball, s.target))
            .map(|s| s.target)
    }

    /// The advance release target — where to strike when there is no shot and no
    /// pass. A kickoff first tries the deflection ladder; otherwise: in genuine
    /// shooting range force a save/rebound at the widest goal-mouth sliver (the
    /// "in the box with nothing on — shoot" rule), else the goal-mouth-cone ray
    /// hoof. Every naive aim goes through [`Self::safe_release`].
    fn advance_target(
        &self,
        world: &World,
        actor: PlayerId,
        ball: Vector2,
        inputs: &PlanInputs,
    ) -> Vector2 {
        // Kickoff with no clear shot: deflect off a distant side opponent — an
        // opponent last-touch gives us the restart deep in their half.
        if inputs.our_attacking_restart && inputs.our_restart_is_kickoff {
            if let Some(t) = self.kickoff_deflect_target(world, ball) {
                return t;
            }
        }
        let opp_goal = world.opp_goal_center();
        let in_finish_zone = ball.x > 0.0 && (ball - opp_goal).norm() < config::STRIKE_FINISH_RANGE;
        let naive = if in_finish_zone {
            // No *clean* window survived the angle gate, but we're in genuine
            // shooting distance: strike the widest sliver (or the centre of a
            // fully crowded mouth) to force a save/rebound/corner rather than
            // giving the ball up — the classic coach call.
            geometry::best_shot_from(world, ball)
                .map(|s| s.target)
                .unwrap_or(opp_goal)
        } else {
            self.hoof_target(world, actor, ball, inputs)
                .unwrap_or(opp_goal)
        };
        self.safe_release(world, ball, naive)
    }

    // ── Release safety ──────────────────────────────────────────────────────

    /// Guard any release aim: keep `naive` if the release is safe, otherwise
    /// steer to the cleanest lane ([`Self::clean_advance`]), otherwise keep
    /// `naive` as the last resort. The single choke point enforcing both "never
    /// kick into a directly shadowing opponent" and "never kick the ball out of
    /// the field" for every hoof, clear, and finish.
    fn safe_release(&self, world: &World, from: Vector2, naive: Vector2) -> Vector2 {
        if self.release_ok(world, from, naive) {
            return naive;
        }
        self.clean_advance(world, from).unwrap_or(naive)
    }

    fn release_ok(&self, world: &World, from: Vector2, target: Vector2) -> bool {
        !self.fires_into_close_opp(world, from, target)
            && !self.exits_field_early(world, from, target)
    }

    /// Veto: would striking from `from` toward `target` fire the ball straight
    /// into a nearby opponent? Tuned very aggressively — the corridor is
    /// collision-sized (a *direct shadow* of the ray only), so an opponent merely
    /// near the lane never vetoes; lane preference is [`Self::clean_advance`]'s
    /// openness scoring, not this.
    fn fires_into_close_opp(&self, world: &World, from: Vector2, target: Vector2) -> bool {
        geometry::fires_into_close_opponent(
            from,
            target,
            world.opp_players(),
            config::CLOSE_OPP_VETO_DIST,
            config::CLOSE_OPP_VETO_CORRIDOR,
        )
    }

    /// Would the release ray leave the field almost immediately (conceding the
    /// ball out)? Rays exiting through the opponent goal mouth are exempt — that
    /// exit is a shot.
    fn exits_field_early(&self, world: &World, from: Vector2, target: Vector2) -> bool {
        let Some(dir) = (target - from).try_normalize(1e-6) else {
            return true;
        };
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let exit = geometry::ray_exit_distance(from, dir, half_len, half_wid);
        if exit >= config::ADVANCE_MIN_TRAVEL {
            return false;
        }
        let exit_pt = from + dir * exit;
        !(exit_pt.x >= half_len - 1.0 && exit_pt.y.abs() <= world.goal_width() / 2.0)
    }

    /// The cleanest forward advancement lane from `from` — the fallback when the
    /// naive release is unsafe. Sweeps a fan of headings from up-field to
    /// pure-lateral (never backward toward our own goal), rejects any that fail
    /// [`Self::release_ok`], and takes the most open lane with a mild
    /// forward-progress tiebreak. `None` only when every heading is blocked or
    /// exits the field immediately.
    fn clean_advance(&self, world: &World, from: Vector2) -> Option<Vector2> {
        let half_len = world.field_length() / 2.0;
        let half_wid = world.field_width() / 2.0;
        let opps = world.opp_players();
        let n = config::ADVANCE_FAN_STEPS;
        let mut best: Option<(Vector2, f64)> = None;
        for i in 0..=n {
            let ang = -config::ADVANCE_FAN_HALF
                + 2.0 * config::ADVANCE_FAN_HALF * (i as f64) / (n as f64);
            let dir = Vector2::new(ang.cos(), ang.sin());
            let exit = geometry::ray_exit_distance(from, dir, half_len, half_wid);
            let target = from + dir * config::ADVANCE_REACH.min(exit);
            if !self.release_ok(world, from, target) {
                continue;
            }
            let open = geometry::lane_openness(from, target, opps, config::ADVANCE_CORRIDOR);
            let score = open + config::ADVANCE_FWD_W * dir.x;
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((target, score));
            }
        }
        best.map(|(t, _)| t)
    }

    // ── Advance aims ────────────────────────────────────────────────────────

    /// Kickoff with no clear shot: aim at a distant opponent off to the side so
    /// the ball deflects off it and runs out of play — an opponent last-touch
    /// gives us the restart deep in their half. Picks the most lateral opponent
    /// among those downfield, distant, and to the side, whose lane from the ball
    /// is clean. `None` (caller falls back to the guarded hoof) when no target
    /// qualifies. No deflection physics is modelled — we just aim at the opponent.
    fn kickoff_deflect_target(&self, world: &World, ball: Vector2) -> Option<Vector2> {
        world
            .opp_players()
            .iter()
            .filter(|o| o.position.x >= config::KICKOFF_DEFLECT_MIN_X)
            .filter(|o| (o.position - ball).norm() >= config::KICKOFF_DEFLECT_MIN_DIST)
            .filter(|o| o.position.y.abs() >= config::KICKOFF_DEFLECT_MIN_LATERAL)
            .filter(|o| !self.fires_into_close_opp(world, ball, o.position))
            .max_by(|a, b| {
                a.position
                    .y
                    .abs()
                    .partial_cmp(&b.position.y.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|o| o.position)
    }

    /// Advancement hoof: score kick *directions* within the cone from the ball to
    /// the opponent goal mouth (see [`geometry::hoof_ray_target`] — travel-
    /// calibration-free, goal-bound so a full-length roll-out can never be a
    /// Div-B aimless kick). Our players already in the opponent half still bias
    /// the ray within the mouth.
    fn hoof_target(
        &self,
        world: &World,
        actor: PlayerId,
        ball: Vector2,
        inputs: &PlanInputs,
    ) -> Option<Vector2> {
        let mates: Vec<Vector2> = world
            .own_players()
            .iter()
            .filter(|p| p.id != actor)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| p.position.x > 0.0)
            .map(|p| p.position)
            .collect();
        geometry::hoof_ray_target(
            ball,
            world.opp_players(),
            &mates,
            world.opp_goal_center().x,
            world.goal_width(),
            geometry::opp_keeper(world),
            config::SHOT_ROBOT_RADIUS,
            geometry::keeper_shot_radius(world),
            config::BALL_RADIUS,
            world.field_width() / 2.0,
            config::HOOF_GOAL_W,
            config::HOOF_MATE_W,
            config::HOOF_MATE_RADIUS,
            config::HOOF_OPEN_PENALTY,
            config::HOOF_RAY_CORRIDOR,
            config::HOOF_CENTER_W,
            config::HOOF_MOUTH_INSET,
        )
    }

    // ── Passes ──────────────────────────────────────────────────────────────

    /// The best forward pass: a supporter clearly ahead of the ball down an open
    /// lane, or — in the final third, where "forward" runs out of field — a wide,
    /// roughly-level cross/cutback outlet. Strictly forward-biased: there is no
    /// recycle, a carrier with nothing on advances the ball itself. Gated by the
    /// global pass-propensity base ([`config::PASS_SUCCESS_BASE`]; 0 disables all
    /// passing). Returns the receiver and the lead target (clamped in-field).
    fn forward_pass(
        &self,
        world: &World,
        actor: PlayerId,
        ball: Vector2,
        inputs: &PlanInputs,
    ) -> Option<(PlayerId, Vector2)> {
        let opp_goal = world.opp_goal_center();
        let opps = world.opp_players();
        let pass_ok = |to: Vector2| -> bool {
            config::PASS_SUCCESS_BASE
                * geometry::lane_openness(ball, to, opps, config::KICK_LANE_CORRIDOR)
                >= config::SUPPORTER_MIN_OPENNESS
        };
        let eligible = |p: &PlayerState| {
            p.id != actor
                && Some(p.id) != inputs.keeper_id
                && Some(p.id) != inputs.double_touch_robot
        };

        // 1. Forward supporter with an open lane from the ball. Score by how far
        //    forward it is, weighted by lane openness.
        let outlet_score = |p: &PlayerState| -> f64 {
            let open = geometry::lane_openness(ball, p.position, opps, config::KICK_LANE_CORRIDOR);
            p.position.x + open * 1000.0
        };
        let mut best = world
            .own_players()
            .iter()
            .filter(|p| eligible(p))
            .filter(|p| p.position.x > ball.x + config::SUPPORTER_FWD_MARGIN)
            .filter(|p| pass_ok(p.position))
            .max_by(|a, b| {
                outlet_score(a)
                    .partial_cmp(&outlet_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        // 2. Final-third cross/cutback: near the opponent goal there is little
        //    field left ahead, so also accept a wide, roughly-level supporter
        //    (laterally separated, not far behind) — the classic cutback outlet.
        if best.is_none() && ball.x > config::FINAL_THIRD_X {
            best = world
                .own_players()
                .iter()
                .filter(|p| eligible(p))
                .filter(|p| p.position.x > ball.x - config::CROSS_BACK_MARGIN)
                .filter(|p| (p.position.y - ball.y).abs() > config::CROSS_MIN_LATERAL)
                .filter(|p| pass_ok(p.position))
                .min_by(|a, b| {
                    let da = (opp_goal - a.position).norm();
                    let db = (opp_goal - b.position).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
        }

        let p = best?;
        // Lead the pass past the receiver toward goal (into space to run onto),
        // shortened along its own direction to stay inside the field.
        let lead = (opp_goal - p.position)
            .try_normalize(1e-6)
            .map(|d| d * config::SUPPORTER_LEAD)
            .unwrap_or_default();
        Some((p.id, clamp_in_field(p.position, p.position + lead, world)))
    }

    /// The corner one-timer (the "1-2" finish). A ball deep and wide of the
    /// opponent box — where `best_shot` never clears [`config::SHOT_MIN_ANGLE`]
    /// and the strike-finish fallback blasts from a near-zero angle — is instead
    /// crossed to a teammate posted at the box edge, which redirects it into the
    /// goal with the pre-armed reflex kick (the coordinator's one-timer mode).
    ///
    /// Gates are pure deflection geometry (the `CORNER_OT_*` consts), never
    /// scaled by [`config::PASS_SUCCESS_BASE`]. Selection is feasibility-first:
    /// pockets and aims are ranked by deflection margin above the coordinator's
    /// `ONE_TIMER_MIN_DEFLECTION` gate (180° = head-on = ideal — its exact
    /// convention), with shot-lane openness only a bounded tiebreak. The receiver
    /// is whichever eligible robot arrives fastest — cross-field pulls are
    /// intended. Returns `None` when the play is off (wrong zone, no feasible
    /// pocket, blocked cross lane, or no receiver in time).
    fn corner_one_timer(
        &self,
        world: &World,
        actor: PlayerId,
        ball: Vector2,
        inputs: &PlanInputs,
    ) -> Option<Waypoint> {
        if !config::CORNER_OT_ENABLED {
            return None;
        }
        let pen = world.opp_penalty_area();
        let pen_half_w = (pen.max.y - pen.min.y) / 2.0;

        // Trigger zone: the BALL deep (near/past the box front) and wide of the
        // box. The caller has already ruled out a real aimed shot.
        let deep = ball.x > pen.min.x - config::CORNER_OT_ZONE_DEPTH;
        let wide = ball.y.abs() > pen_half_w + config::CORNER_OT_MIN_WIDE;
        if !deep || !wide {
            return None;
        }

        let opps = world.opp_players();
        let opp_goal = world.opp_goal_center();
        // Aim candidates stay off the posts by this much (ball + slack).
        let aim_inset = world.goal_width() / 2.0 - 120.0;

        // Deflection angle (degrees) at `pocket` redirecting the incoming cross
        // toward `aim` — the coordinator's exact formula (pass_coordinator.rs):
        // 180° = head-on return = ideal, below its gate the ball arrives behind
        // the kicker mouth.
        let deflection = |pocket: Vector2, aim: Vector2| -> f64 {
            (pocket - ball).angle(&(aim - pocket)).to_degrees()
        };

        // Best redirect aim from `pocket`: feasibility (deflection margin over
        // the commit bar) is the primary pull; openness of the pocket→goal lane
        // adds at most CORNER_OT_AIM_OPENNESS_WEIGHT degrees of equivalent
        // margin. Returns `None` when no goal-mouth aim is feasible.
        let best_aim = |pocket: Vector2| -> Option<(Vector2, f64)> {
            const AIM_SAMPLES: usize = 5;
            let mut best: Option<(Vector2, f64)> = None;
            for i in 0..AIM_SAMPLES {
                let t = i as f64 / (AIM_SAMPLES - 1) as f64;
                let aim = Vector2::new(opp_goal.x, -aim_inset + t * 2.0 * aim_inset);
                let margin = deflection(pocket, aim) - config::CORNER_OT_MIN_DEFLECTION;
                if margin < 0.0 {
                    continue;
                }
                let open = geometry::lane_openness(pocket, aim, opps, config::KICK_LANE_CORRIDOR);
                let score = margin + open * config::CORNER_OT_AIM_OPENNESS_WEIGHT;
                if best.map_or(true, |(_, s)| score > s) {
                    best = Some((aim, score));
                }
            }
            best
        };

        // Pocket candidates hug the defense-area edge (standoff keeps the touch
        // point legal): the front edge plus both side edges. Near-side pockets
        // self-filter — their deflection to goal is far below the gate.
        let standoff = config::CORNER_OT_POCKET_MARGIN;
        let mut pockets: Vec<Vector2> = Vec::with_capacity(11);
        const FRONT_SAMPLES: usize = 5;
        for i in 0..FRONT_SAMPLES {
            let t = i as f64 / (FRONT_SAMPLES - 1) as f64;
            let y = -pen_half_w + t * 2.0 * pen_half_w;
            pockets.push(Vector2::new(pen.min.x - standoff, y));
        }
        const SIDE_SAMPLES: usize = 3;
        let depth = pen.max.x - pen.min.x;
        for i in 0..SIDE_SAMPLES {
            let x = pen.min.x + depth * (i as f64 + 0.5) / SIDE_SAMPLES as f64;
            pockets.push(Vector2::new(x, pen_half_w + standoff));
            pockets.push(Vector2::new(x, -(pen_half_w + standoff)));
        }

        // Rank pockets: feasible aim (hard), staging room behind the ball on
        // the cross axis (hard — the coordinator drives through the ball from
        // behind, and a deep-corner ball puts that point outside the field for
        // out-pointing axes), open cross lane from the ball (hard), opponent
        // sitting on the pocket discounts but never vetoes — adjacent samples
        // win, the "slide along the box edge" behavior.
        let half_len = world.field_length() / 2.0 + world.field_boundary_width();
        let half_wid = world.field_width() / 2.0 + world.field_boundary_width();
        let mut best_pocket: Option<(Vector2, Vector2, f64)> = None;
        for pocket in pockets {
            let Some(axis) = (ball - pocket).try_normalize(1e-6) else {
                continue;
            };
            let stage = ball + axis * config::CORNER_OT_STAGE_ROOM;
            // Robot-centre clearance to the boundary lines (radius + slack).
            if stage.x.abs() > half_len - 100.0 || stage.y.abs() > half_wid - 100.0 {
                continue;
            }
            let lane = geometry::lane_openness(ball, pocket, opps, config::KICK_LANE_CORRIDOR);
            if lane < config::CORNER_OT_MIN_LANE {
                continue;
            }
            let Some((aim, aim_score)) = best_aim(pocket) else {
                continue;
            };
            let clear = config::CORNER_OT_CLEARANCE_FLOOR
                + (1.0 - config::CORNER_OT_CLEARANCE_FLOOR)
                    * geometry::receiver_clearance(pocket, opps);
            let score = aim_score * lane * clear;
            if best_pocket.map_or(true, |(_, _, s)| score > s) {
                best_pocket = Some((pocket, aim, score));
            }
        }
        let (pocket, aim, _) = best_pocket?;

        // Nominate the fastest eligible receiver to the pocket. The travel cap
        // keeps the commit inside the coordinator's stage budget with margin.
        let receiver = world
            .own_players()
            .iter()
            .filter(|p| p.id != actor)
            .filter(|p| Some(p.id) != inputs.keeper_id)
            .filter(|p| Some(p.id) != inputs.double_touch_robot)
            .filter(|p| !crate::ball_duty_barred(p))
            .map(|p| {
                let t = geometry::redirect_time(
                    p.position,
                    p.velocity,
                    pocket,
                    config::V_MAX,
                    config::A_MAX,
                );
                (p.id, t)
            })
            .filter(|(_, t)| *t <= config::CORNER_OT_MAX_TRAVEL)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| id)?;

        debug::value("corner_ot.deflection", deflection(pocket, aim));
        debug::cross("corner_ot.pocket", pocket);

        Some(Waypoint::Pass {
            passer: actor,
            receiver,
            target_area: pocket,
            forward_to: Some(aim),
        })
    }
}

/// Pull a pass lead target inside the field, preserving the lead direction. The
/// lead aims a pass into space ahead of a receiver (toward the opponent goal);
/// for a deep or wide receiver that space can fall past the goal line or a
/// touchline, which would send the ball out. We shorten the lead along its own
/// direction so the aim point lands [`config::FIELD_LEAD_MARGIN`] inside the
/// boundary, rather than axis-clamping it (which would skew the aim sideways).
fn clamp_in_field(from: Vector2, target: Vector2, world: &World) -> Vector2 {
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

    fn world_at(ball: Vector2, own: Vec<PlayerState>, opp: Vec<PlayerState>) -> World {
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
            double_touch_barred: None,
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
            our_restart_is_kickoff: false,
        }
    }

    fn strike_target(plan: &Plan) -> Vector2 {
        match &plan.waypoints[0] {
            Waypoint::Strike { target } => *target,
            other => panic!("expected a Strike, got {other:?}"),
        }
    }

    /// The chosen release ray must clear `marker` by more than a collision width.
    fn assert_clears(ball: Vector2, target: Vector2, marker: Vector2) {
        let dir = (target - ball).normalize();
        let to_m = marker - ball;
        let perp = (to_m.x * dir.y - to_m.y * dir.x).abs();
        let along = to_m.dot(&dir);
        assert!(
            along <= 0.0
                || along > config::CLOSE_OPP_VETO_DIST
                || perp > config::CLOSE_OPP_VETO_CORRIDOR,
            "release must not fire into the marker: along={along:.0} perp={perp:.0} target={target:?}"
        );
    }

    #[test]
    fn freekick_in_our_half_steers_around_a_close_marker() {
        // Regression (live log t=1125): a free kick deep in our half with a single
        // close marker and the whole field open elsewhere must NOT be hoofed
        // straight into the marker — it steers down a clean lane around it.
        let ball = Vector2::new(-1989.0, -2549.0);
        let own = vec![
            player(1, -4000.0, 0.0),     // keeper
            player(2, -2371.0, -2695.0), // kicker, on the ball
        ];
        let marker = Vector2::new(-1790.0, -2500.0); // ~205mm away, ahead on the lane
        let opp = vec![
            PlayerState::new(
                PlayerId::new(2),
                marker,
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            ),
            player(12, 2660.0, -169.0), // rest of the team far up-field
            player(3, 3326.0, -471.0),
            player(14, 4348.0, -36.0), // their keeper (deepest)
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true; // free kick (not kickoff)

        let plan = planner
            .replan(&world, &Possession::Loose, Some(PlayerId::new(2)), &inp)
            .expect("should produce a plan");
        assert_clears(ball, strike_target(&plan), marker);
    }

    #[test]
    fn attacking_restart_kicker_shoots_when_goal_clear() {
        // A restart kicker in range with a genuinely clear mouth strikes directly
        // at goal — held to the same "actually clear" gate as open play.
        let ball = Vector2::new(-1000.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -1200.0, 0.0)];
        let world = world_at(ball, own, vec![]); // empty mouth
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true;

        let plan = planner
            .replan(&world, &Possession::Loose, Some(PlayerId::new(2)), &inp)
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            (target.x - world.field_length() / 2.0).abs() < 1e-6
                && target.y.abs() < world.goal_width() / 2.0,
            "clear mouth: restart strike should aim into the goal, got {target:?}"
        );
    }

    #[test]
    fn attacking_restart_freekick_advances_when_not_clear() {
        // A free kick with a keeper guarding the mouth (only sub-gate slivers
        // open) must NOT take the hopeful shot: it advances forward instead —
        // the goal-mouth hoof exits through their back line.
        let ball = Vector2::new(-1000.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -1200.0, 0.0)];
        let opp = vec![player(9, 4400.0, 0.0)]; // lone keeper closing the mouth
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true;

        let plan = planner
            .replan(&world, &Possession::Loose, Some(PlayerId::new(2)), &inp)
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            (target.x - world.field_length() / 2.0).abs() < 1e-6,
            "no clear shot: free kick must advance forward through their back line, got {target:?}"
        );
    }

    #[test]
    fn kickoff_with_no_clear_shot_deflects_off_a_side_opponent() {
        // Regression: an opponent lined up in front of the kickoff must NOT be
        // struck straight; the kick deflects off a distant side opponent instead
        // (an opponent last-touch gives us the restart deep in their half).
        let ball = Vector2::new(-1000.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -1200.0, 0.0)];
        let opp = vec![
            player(9, 4400.0, 0.0),     // opp keeper closing the mouth
            player(8, -700.0, 0.0),     // opponent lined up dead ahead of the ball
            player(7, 1500.0, 2200.0),  // distant side opponent (+y)
            player(6, 1500.0, -1800.0), // distant side opponent (−y, less lateral)
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();
        let mut inp = inputs();
        inp.our_attacking_restart = true;
        inp.our_restart_is_kickoff = true;

        let plan = planner
            .replan(&world, &Possession::Loose, Some(PlayerId::new(2)), &inp)
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            (target - Vector2::new(1500.0, 2200.0)).norm() < 1e-6,
            "kickoff must deflect off the more-lateral side opponent, got {target:?}"
        );
    }

    #[test]
    fn open_forward_supporter_draws_a_pass() {
        // Carrier deep in our half (no shot in range), a clear forward supporter,
        // no opponents in the lane → a coordinated pass, never a recycle.
        let ball = Vector2::new(-1000.0, 0.0);
        let own = vec![
            player(1, -4000.0, 0.0), // keeper
            player(2, -1000.0, 0.0), // carrier
            player(3, 1500.0, 0.0),  // open forward supporter
        ];
        // A body in the mouth so the empty-field clear-shot gate doesn't fire.
        let opp = vec![player(9, 4400.0, 0.0)];
        let world = world_at(ball, own, opp);
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
    fn stable_opp_possession_is_harassed() {
        // Opponent holding the ball at midfield → the elected challenger parks in
        // front of its dribbler (Harass), it does not chase the ball sideways.
        let ball = Vector2::new(500.0, 300.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -500.0, 300.0)];
        let opp = vec![player(9, 550.0, 300.0)]; // the holder
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(
                &world,
                &Possession::Opp(PlayerId::new(9)),
                Some(PlayerId::new(2)),
                &inputs(),
            )
            .expect("should produce a plan");
        assert_eq!(
            plan.waypoints[0],
            Waypoint::Harass {
                opponent: PlayerId::new(9)
            }
        );
        assert_eq!(plan.active_robot, PlayerId::new(2));
    }

    #[test]
    fn opp_possession_in_the_wall_apron_is_pursued_not_harassed() {
        // The holder is deep in front of our goal (the wall's apron — outside
        // the keeper's box): a harasser parked on it would screen our own
        // wall/keeper, so the challenger pursues the ball itself (a plain
        // Strike clear).
        let ball = Vector2::new(-2800.0, 200.0);
        let own = vec![player(1, -4300.0, 0.0), player(2, -2300.0, 300.0)];
        let opp = vec![player(9, -2850.0, 200.0)];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(
                &world,
                &Possession::Opp(PlayerId::new(9)),
                Some(PlayerId::new(2)),
                &inputs(),
            )
            .expect("should produce a plan");
        assert!(
            matches!(plan.waypoints[0], Waypoint::Strike { .. }),
            "in the apron the challenger strikes the ball clear, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn box_carrier_with_no_clean_shot_strikes() {
        // Carrier on the ball in the box, the goal mouth crowded so no *clean*
        // aimed-shot window survives the angle gate, and no forward outlet: the
        // carrier releases a one-touch Strike at the goal (force a save/rebound),
        // never a backward ball.
        let ball = Vector2::new(3700.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, 3700.0, 0.0)];
        let opp = vec![
            player(7, 4200.0, -300.0),
            player(8, 4200.0, 0.0),
            player(9, 4200.0, 300.0),
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            target.x > ball.x,
            "the box finish must go goalward, got {target:?}"
        );
    }

    #[test]
    fn deep_carrier_with_no_options_hoofs_through_their_back_line() {
        // Crowded mouth, carrier in its own half, no supporters: the release
        // falls through to the goal-mouth-cone ray hoof (whose failure
        // distribution from deep is benign) and exits through their back line.
        let ball = Vector2::new(-1500.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -1500.0, 0.0)];
        let opp = vec![
            player(7, 4200.0, -300.0),
            player(8, 4200.0, 0.0),
            player(9, 4200.0, 300.0),
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            (target.x - world.field_length() / 2.0).abs() < 1e-6,
            "deep release must hoof through their back line, got {target:?}"
        );
    }

    #[test]
    fn loose_ball_gets_the_same_ladder_as_a_held_one() {
        // A loose ball in the strike zone is struck first-time by the capturer —
        // one uninterrupted strike-through episode, same decision as a carrier.
        let ball = Vector2::new(2600.0, 200.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, 2400.0, 200.0)];
        let opp = vec![player(9, 3800.0, 0.0)];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(
                &world,
                &Possession::Loose,
                Some(PlayerId::new(2)),
                &inputs(),
            )
            .expect("should produce a plan");
        assert!(matches!(plan.waypoints[0], Waypoint::Strike { .. }));
        assert_eq!(plan.active_robot, PlayerId::new(2));
    }

    #[test]
    fn touchline_ball_is_struck_inward_not_out() {
        // A ball pinned on a touchline with its natural forward lane shadowed by
        // a marker: the release must neither hit the marker nor leave the field
        // almost immediately — the safe-release gate steers it to an in-field lane.
        let half_wid = 3000.0;
        let ball = Vector2::new(500.0, half_wid - 60.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, 300.0, half_wid - 300.0)];
        // Marker directly shadowing the ball→goal-mouth ray.
        let marker = Vector2::new(720.0, half_wid - 220.0);
        let opp = vec![
            PlayerState::new(
                PlayerId::new(9),
                marker,
                Vector2::new(0.0, 0.0),
                Angle::from_radians(0.0),
            ),
            player(8, 4400.0, 0.0), // keeper in the mouth
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(
                &world,
                &Possession::Loose,
                Some(PlayerId::new(2)),
                &inputs(),
            )
            .expect("should produce a plan");
        let target = strike_target(&plan);
        let dir = (target - ball).normalize();
        let exit = geometry::ray_exit_distance(
            ball,
            dir,
            world.field_length() / 2.0,
            world.field_width() / 2.0,
        );
        let exit_pt = ball + dir * exit;
        let through_mouth = exit_pt.x >= world.field_length() / 2.0 - 1.0
            && exit_pt.y.abs() <= world.goal_width() / 2.0;
        assert!(
            through_mouth || exit >= config::ADVANCE_MIN_TRAVEL,
            "release must not concede the ball out: exit={exit:.0} target={target:?}"
        );
        assert_clears(ball, target, marker);
    }

    #[test]
    fn veto_requires_a_direct_shadow() {
        // An opponent near the release lane but NOT directly shadowing the ray
        // must not veto it (the aggressive direct-shadow tuning): the hoof keeps
        // its natural goal-mouth aim.
        let ball = Vector2::new(-1500.0, 0.0);
        let own = vec![player(1, -4000.0, 0.0), player(2, -1700.0, 0.0)];
        let opp = vec![
            // ~400mm ahead but ~350mm off the lane: close, not a direct shadow.
            player(8, -1100.0, 350.0),
            player(9, 4400.0, 0.0), // keeper in the mouth
        ];
        let world = world_at(ball, own, opp);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        let target = strike_target(&plan);
        assert!(
            (target.x - world.field_length() / 2.0).abs() < 1e-6 && target.y.abs() < 500.0,
            "a near-but-clear opponent must not veto the mouth hoof, got {target:?}"
        );
    }

    // ── Corner one-timer battery ────────────────────────────────────────────

    /// Div-B corner geometry shared by the corner one-timer tests: carrier on
    /// the ball deep in the opponent corner, their keeper guarding the (already
    /// razor-thin) ball→mouth cone so no direct-shot window clears the angle
    /// gate, a presser behind the ball. From (4150, 2500) the whole mouth
    /// subtends only ~0.057 rad; the keeper at (4400, 700) sits inside that
    /// cone and splits it into sub-gate slivers, so the ladder falls through
    /// to the corner rung instead of blasting the sliver.
    fn corner_world(own: Vec<PlayerState>, extra_opps: Vec<PlayerState>) -> World {
        let ball = Vector2::new(4150.0, 2500.0);
        let mut opp = vec![
            player(14, 4400.0, 700.0), // their keeper, on the ball→mouth cone
            player(5, 4300.0, 2500.0), // presser behind the ball
        ];
        opp.extend(extra_opps);
        world_at(ball, own, opp)
    }

    #[test]
    fn corner_carrier_emits_one_timer_pass() {
        // The 1-2 finish: carrier in the opponent corner (shot angle far below
        // SHOT_MIN_ANGLE) with a lone teammate on the other side of the field
        // must cross to a box-edge pocket for a reflex redirect into the goal.
        if !config::CORNER_OT_ENABLED {
            return; // play switched off — nothing to assert
        }
        let own = vec![
            player(1, -4000.0, 0.0),    // keeper
            player(2, 4150.0, 2500.0),  // carrier, on the ball in the corner
            player(3, 1500.0, -1500.0), // far-side teammate — the pull
        ];
        let world = corner_world(own, vec![]);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        match &plan.waypoints[0] {
            Waypoint::Pass {
                passer,
                receiver,
                target_area,
                forward_to,
            } => {
                assert_eq!(*passer, PlayerId::new(2));
                assert_eq!(*receiver, PlayerId::new(3), "the lone teammate is pulled");
                let aim = forward_to.expect("a corner pass must be a one-timer");
                assert!(
                    (aim.x - 4500.0).abs() < 1e-6 && aim.y.abs() <= 500.0,
                    "redirect must aim into the goal mouth, got {aim:?}"
                );
                let pen = world.opp_penalty_area();
                assert!(
                    !pen.contains(*target_area),
                    "receive pocket must be outside the defense area, got {target_area:?}"
                );
                let ball = world.ball_position().unwrap();
                let deflection = (target_area - ball)
                    .angle(&(aim - target_area))
                    .to_degrees();
                assert!(
                    deflection >= config::CORNER_OT_MIN_DEFLECTION,
                    "deflection {deflection:.0}° below the commit bar"
                );
            }
            other => panic!("expected the corner one-timer Pass, got {other:?}"),
        }
    }

    #[test]
    fn corner_one_timer_needs_a_receiver() {
        // Nobody to pull (keeper is never eligible): the corner carrier falls
        // back to a first-time strike, never a pass.
        let own = vec![
            player(1, -4000.0, 0.0),   // keeper
            player(2, 4150.0, 2500.0), // carrier, on the ball in the corner
        ];
        let world = corner_world(own, vec![]);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            !matches!(&plan.waypoints[0], Waypoint::Pass { .. }),
            "no eligible receiver: must not pass, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn corner_one_timer_blocked_cross_lanes_fall_through() {
        // A corner trap: an opponent arc between the ball and every box-edge
        // pocket kills the cross-lane gate — the play stays off and the carrier
        // strikes first-time instead of feeding the wall.
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 4150.0, 2500.0),
            player(3, 1500.0, -1500.0),
        ];
        // Ring at ~600mm from the ball spanning the bearings toward the box.
        let trap = vec![
            player(6, 4098.0, 2052.0),
            player(7, 3806.0, 2159.0),
            player(8, 3606.0, 2396.0),
        ];
        let world = corner_world(own, trap);
        let mut planner = Planner::new();

        let plan = planner
            .replan(&world, &Possession::We(PlayerId::new(2)), None, &inputs())
            .expect("should produce a plan");
        assert!(
            !matches!(&plan.waypoints[0], Waypoint::Pass { .. }),
            "every cross lane is walled: must not pass, got {:?}",
            plan.waypoints[0]
        );
    }

    #[test]
    fn corner_one_timer_zone_gates() {
        // Wide but not deep, and deep but not wide (a workable strike angle):
        // neither position is the corner play.
        let planner = Planner::new();
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 2400.0, 2650.0), // wide midfielder, not deep
            player(3, 1500.0, -1500.0),
        ];
        let ball = Vector2::new(2400.0, 2650.0);
        let world = world_at(ball, own, vec![]);
        assert!(
            planner
                .corner_one_timer(&world, PlayerId::new(2), ball, &inputs())
                .is_none(),
            "not deep enough: zone gate must hold"
        );

        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 4000.0, 800.0), // deep but in front of the box
            player(3, 1500.0, -1500.0),
        ];
        let ball = Vector2::new(4000.0, 800.0);
        let world = world_at(ball, own, vec![]);
        assert!(
            planner
                .corner_one_timer(&world, PlayerId::new(2), ball, &inputs())
                .is_none(),
            "not wide of the box: zone gate must hold"
        );
    }

    #[test]
    fn corner_one_timer_needs_staging_room() {
        // Ball hard against the touchline: every cross axis out of the corner
        // leaves no in-field room for the passer to stage behind the ball, so
        // the play stays off (the release would wedge on the boundary).
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 4200.0, 2850.0),
            player(3, 1500.0, -1500.0),
        ];
        let ball = Vector2::new(4200.0, 2850.0);
        let world = world_at(ball, own, vec![]);
        let planner = Planner::new();
        assert!(
            planner
                .corner_one_timer(&world, PlayerId::new(2), ball, &inputs())
                .is_none(),
            "no staging room behind a touchline ball: play must stay off"
        );
    }

    #[test]
    fn corner_one_timer_skips_barred_receiver() {
        // A receiver needs a working kicker for the reflex redirect: the closer
        // teammate with a dead kicker is skipped for the farther healthy one.
        if !config::CORNER_OT_ENABLED {
            return; // play switched off — nothing to assert
        }
        let mut near = player(3, 2800.0, -800.0);
        near.handicaps.insert(Handicap::NoKicker);
        let own = vec![
            player(1, -4000.0, 0.0),
            player(2, 4150.0, 2500.0),
            near,
            player(4, 1500.0, -1500.0),
        ];
        let world = corner_world(own, vec![]);
        let planner = Planner::new();
        let ball = world.ball_position().unwrap();

        match planner.corner_one_timer(&world, PlayerId::new(2), ball, &inputs()) {
            Some(Waypoint::Pass { receiver, .. }) => {
                assert_eq!(receiver, PlayerId::new(4), "NoKicker robot must be skipped");
            }
            other => panic!("expected the one-timer Pass, got {other:?}"),
        }
    }
}
