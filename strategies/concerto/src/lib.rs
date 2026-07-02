//! Concerto — a formation/planner-split RoboCup SSL strategy.
//!
//! Two layers above the skills:
//! - **Formation** runs one cost-aware matching that positions every field robot
//!   (except the keeper and the plan-controlled robots) and, when we don't hold the
//!   ball, elects the ball-winner as one of those roles.
//! - **Planner → Driver** moves the ball toward the opponent goal via
//!   ball-state-transition waypoints, re-deciding only on discrete events.

pub mod config;
pub mod driver;
pub mod formation;
pub mod geometry;
pub mod keeper;
pub mod logo;
pub mod matching;
pub mod planner;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use driver::{Driver, WaypointStatus};
use formation::Formation;
use planner::{PlanInputs, Planner};

/// Standoff distance (mm) the set-piece kicker stages behind the ball during a
/// prepare/setup state. The executor *hard-enforces* a no-touch ball keep-out for
/// our kicker in `PrepareKickoff` (no robot may touch the ball before Normal Start
/// — rules §5.3.2); this places the staging target just behind the ball on the
/// attacking axis so the kicker sits poised inside the center circle to pounce.
const PREP_KICKER_STANDOFF: f64 = 300.0;

/// The Concerto strategy.
pub struct ConcertoStrategy {
    /// Previous frame's possession, for detecting changes (the metric itself is
    /// the framework's single source of truth — `World::possession`).
    last_possession: Possession,
    /// Previous frame's contest activity, so contest onset/clearing triggers a
    /// replan even while `possession` is unchanged (it stays `We(id)` when an
    /// opponent starts pressing the ball we hold).
    last_contest_active: bool,
    planner: Planner,
    driver: Driver,
    formation: Formation,
    last_game_state: GameState,
    double_touch_robot: Option<PlayerId>,
    /// The pinned taker of our current set piece. Latched at designation (prep
    /// or first live election) and held until the restart resolves — the kick
    /// is released, another robot takes it, or the robot leaves the field — so
    /// the kicker identity can never flicker mid-take.
    restart_kicker: Option<PlayerId>,
    /// Ball position when we gained possession (the dribble contact point), for the
    /// excessive-dribbling carry cap. Cleared whenever we don't hold the ball.
    dribble_origin: Option<Vector2>,
    /// World time we gained possession, for the transient fast-break window. Cleared
    /// whenever we don't hold the ball, so it re-stamps on the next turnover.
    gained_ball_t: Option<f64>,
    /// Goalkeeper Guard/Clear state machine.
    keeper: keeper::KeeperState,
    /// Wall robot currently executing a reflex strike on an incoming ball (see
    /// `config::WALL_STRIKE_ENABLED`). While set, it is a reserved plan slot
    /// (its own body holds its wall slot via the Shadow slot-hold) and the open
    /// capture role is suppressed so nobody chases the ball into the wall.
    wall_striker: Option<PlayerId>,
    /// Robots serving Shadow roles last tick (from Formation's commands) — the
    /// wall-striker candidate pool. One tick stale by construction, which is fine
    /// at frame rate.
    last_wall_ids: Vec<PlayerId>,
}

impl ConcertoStrategy {
    pub fn new() -> Self {
        Self {
            last_possession: Possession::Loose,
            last_contest_active: false,
            planner: Planner::new(),
            driver: Driver::new(),
            formation: Formation::new(),
            last_game_state: GameState::Unknown,
            double_touch_robot: None,
            restart_kicker: None,
            dribble_origin: None,
            gained_ball_t: None,
            keeper: keeper::KeeperState::new(),
            wall_striker: None,
            last_wall_ids: Vec::new(),
        }
    }
}

impl Default for ConcertoStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl Strategy for ConcertoStrategy {
    fn init(&mut self, _world: &World) {
        tracing::info!("Concerto strategy initialized");
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::bool("defense_only", "Defense only (suppress offense)", false),
            ParamSpec::bool(
                "warmup",
                "Warmup: pose in the logo (triangle) formation",
                false,
            ),
        ]
    }

    fn update(&mut self, ctx: &mut TeamContext) {
        let world = World::new(ctx.world().raw_snapshot().clone());
        let game_state = world.game_state();
        let us_operating = world.us_operating();
        // During a stoppage the executor may present the *predicted* upcoming
        // restart (free kick / kickoff / penalty) here so we can pre-stage for
        // it. `pre_stage` is true in that window: stage positions, but don't run
        // live play — the real stoppage rules are still enforced by the executor.
        let pre_stage = world.pre_stage();
        // Replay clarity: the executor/GC log shows the real state (e.g. Stop)
        // while we're acting on the predicted restart shown here.
        debug::value("pre_stage", if pre_stage { 1.0 } else { 0.0 });
        if pre_stage {
            debug::string(
                "pre_stage_restart",
                &format!("{:?} us_operating={}", game_state, us_operating),
            );
        } else {
            debug::remove("pre_stage_restart");
        }
        let now = world.timestamp();
        let defense_only = ctx.param_bool("defense_only");

        // ── Double-touch tracking ───────────────────────────────────────
        // Keyed off the framework's *bar* (the taker released the ball into
        // play), not the first-touch identity latch — barring on identity would
        // evict the taker from its own kick the moment it legally touches.
        if let Some(barred) = world.double_touch_barred() {
            self.double_touch_robot = Some(barred);
        } else if matches!(
            game_state,
            GameState::Halt | GameState::Stop | GameState::Timeout | GameState::Run
        ) {
            // Reset on stoppages, and on Run once the framework clears the kicker
            // (another robot has touched the ball).
            self.double_touch_robot = None;
        }

        // ── Restart-kicker pin ──────────────────────────────────────────
        // While our set piece is being taken, the taker's identity is frozen:
        // latched at designation, released only when the restart resolves.
        // During `pre_stage` the shown `game_state`/`us_operating` already
        // reflect the predicted restart, so the window spans prep → live.
        let take_window = us_operating
            && matches!(
                game_state,
                GameState::PrepareKickoff
                    | GameState::Kickoff
                    | GameState::FreeKick
                    | GameState::PreparePenalty
                    | GameState::Penalty
                    | GameState::PenaltyRun
            );
        if !take_window {
            self.restart_kicker = None;
        } else if let Some(pin) = self.restart_kicker {
            // Release when: the taker released the ball into play (pin→bar
            // handoff — the bar then keeps it off the ball), a different own
            // robot touched first (interference/steal of the take), or the
            // pinned robot left the field (sidelined/carded → re-latch).
            let taken = world.double_touch_barred() == Some(pin);
            let other_took = matches!(world.freekick_kicker(), Some(k) if k != pin);
            let on_field = world.own_players().iter().any(|p| p.id == pin);
            if taken || other_took || !on_field {
                self.restart_kicker = None;
            }
        }

        // ── Game-state transition → clear plan + driver ─────────────────
        let game_state_changed = game_state != self.last_game_state;
        if game_state_changed {
            self.planner.clear_plan();
            self.driver.clear();
        }
        self.last_game_state = game_state;

        // ── Logo (triangle) formation ───────────────────────────────────
        // Active on the operator's `warmup` toggle (pose up while placing robots),
        // or during a real-match Timeout (cosmetic ready-formation). Robots tagged
        // `formation` are exempt from the executor's stop-state speed clamp (except
        // Halt), so they may slowly reposition. This takes precedence over normal
        // play, so leaving `warmup` on is an obvious visible signal.
        let warmup = ctx.param_bool("warmup");
        let logo_active = warmup || matches!(game_state, GameState::Timeout);
        if logo_active {
            self.run_logo(&world, ctx);
            return;
        }

        // ── Hard stops: let the executor halt the robots ────────────────
        if matches!(game_state, GameState::Halt | GameState::Unknown) {
            return;
        }

        let possession = world.possession();
        let possession_changed = possession != self.last_possession;
        self.last_possession = possession;

        // Contest onset/clearing is its own replan trigger: when an opponent starts
        // pressing the ball we hold, `possession` stays `We(id)` (breakbeam latches
        // ownership) so nothing else would prompt a re-decision — and the pinned
        // dribble would only fail slowly on timeout. Edge-detect it here.
        let contest_active = world.ball_contest().is_some();
        let contest_changed = contest_active != self.last_contest_active;
        self.last_contest_active = contest_active;

        // Track the dribble contact point: stamp it when we gain the ball, clear it
        // when we don't hold it. `carried` = how far we've dribbled since contact.
        match possession {
            Possession::We(_) => {
                if self.dribble_origin.is_none() {
                    self.dribble_origin = world.ball_position();
                }
                if self.gained_ball_t.is_none() {
                    self.gained_ball_t = Some(now);
                }
            }
            _ => {
                self.dribble_origin = None;
                self.gained_ball_t = None;
            }
        }
        let carried = match (self.dribble_origin, world.ball_position()) {
            (Some(origin), Some(ball)) => (ball - origin).norm(),
            _ => 0.0,
        };

        let ball_present = world.ball_position().is_some();
        let we_may_act = match game_state {
            GameState::Run => true,
            GameState::Kickoff | GameState::FreeKick | GameState::PenaltyRun => us_operating,
            _ => false,
        };

        // `!pre_stage` keeps us from running live play while the executor is only
        // showing us a *predicted* restart during a stoppage (line-up only). This
        // only bites our own free kick — kickoff/penalty predictions surface as
        // `Prepare*` states, which aren't ball-in-play, so `can_act` is already
        // false for them.
        let can_act =
            !defense_only && ball_present && world.is_ball_in_play() && we_may_act && !pre_stage;
        // A pass is atomic: possession flits We→Loose→We while the ball is in flight,
        // so we stay in the offense path (the coordinator owns the pass) rather than
        // treating the transient loose ball as something to chase.
        let passing = self.driver.is_passing();
        // The keeper holding the ball is neither offense nor pursuit: its Guard/Clear
        // machine owns it (it will strike the ball out itself), the planner must not
        // build a carrier plan around it, and no field robot should chase a ball our
        // keeper holds inside the box.
        let keeper_has_ball =
            matches!(possession, Possession::We(id) if Some(id) == world.our_keeper_id());
        // Offense — we hold the ball (or a pass is in flight): the Planner owns the
        // active robot(s). Pursuit — a ball we don't hold but may contest: Formation's
        // matching picks the capturer, weighing it against every defensive duty.
        let offense =
            can_act && !keeper_has_ball && (passing || matches!(possession, Possession::We(_)));
        let pursuit = can_act && !offense && !keeper_has_ball;

        let inputs = PlanInputs {
            keeper_id: world.our_keeper_id(),
            double_touch_robot: self.double_touch_robot,
            our_attacking_restart: us_operating
                && matches!(game_state, GameState::Kickoff | GameState::FreeKick),
            carried,
            fresh_possession: self
                .gained_ball_t
                .map(|t| now - t < config::FAST_BREAK_WINDOW)
                .unwrap_or(false),
            now,
        };

        // Robots Formation must not position: those the Planner already controls
        // (carrier + pass receiver). The pursuit capturer is *not* reserved — it is
        // chosen by Formation's matching and excluded from positioning there.
        let mut reserved: Vec<PlayerId> = Vec::new();

        // ── Offense: plan on events (Planner-first so pass slots are reserved) ──
        if offense {
            let needs_replan = self.planner.current_plan().is_none()
                || matches!(
                    self.driver.status(),
                    WaypointStatus::Succeeded | WaypointStatus::Failed(_)
                )
                || (!passing && (possession_changed || contest_changed || game_state_changed));
            if needs_replan {
                match self.planner.replan(&world, &possession, None, &inputs) {
                    Some(plan) => self
                        .driver
                        .set_waypoint(plan.waypoints[0].clone(), plan.active_robot),
                    None => self.driver.clear(),
                }
            }
            if self.driver.active_robot_id().is_some() {
                reserved.extend(self.driver.plan_slots());
                // On a Failed status the next tick re-derives a fresh plan (same
                // most-eligible robot re-engages); no per-robot failure memory.
                self.driver.update(&world, ctx);
                // A completed pass hands the ball to the receiver; the Succeeded
                // status triggers the replan that picks up the new carrier.
                let _ = self.driver.take_new_active();
            }
        } else if !pursuit {
            // Not acting — replan fresh when play resumes.
            self.planner.clear_plan();
            self.driver.clear();
        }

        // ── Our set-piece preparation: designate & position a kicker ─────
        // Covers states that are not yet ball-in-play but where comply() would
        // otherwise clamp/sideline our kicker: PrepareKickoff, PreparePenalty, and
        // the brief Penalty state before PenaltyRun. Also covers a *predicted* free
        // kick during a stoppage (`pre_stage`): there is no prepare-variant for
        // free kicks, so we line the taker up here instead of running live offense.
        if us_operating
            && (matches!(
                game_state,
                GameState::PrepareKickoff | GameState::PreparePenalty | GameState::Penalty
            ) || (pre_stage && game_state == GameState::FreeKick))
        {
            if let Some(id) = self.designate_prep_kicker(&world, ctx, game_state) {
                reserved.push(id);
            }
        }

        // ── Wall reflex strike: a ball rolled/kicked into our wall is met by the
        //    wall robot it is arriving at — a one-touch strike straight forward
        //    through the free ball, never held — instead of an outside capturer
        //    stern-chasing the ball into the wall corridor. Exit first (episode
        //    resolved / geometry lapsed / another mode owns the driver), then
        //    arm on a fresh incoming ball. ──────────────────────────────────────
        if let Some(sid) = self.wall_striker {
            if !pursuit {
                // Offense or a stoppage owns the driver now (it was replanned or
                // cleared above); just drop the strike designation.
                self.wall_striker = None;
            } else if matches!(
                self.driver.status(),
                WaypointStatus::Succeeded | WaypointStatus::Failed(_)
            ) || !wall_strike_valid(&world, sid)
            {
                // A Failed whiff clears here; if the ball is still incoming the
                // entry below re-arms next tick (always re-engage, never rotate).
                self.wall_striker = None;
                self.driver.clear();
            }
        }
        if config::WALL_STRIKE_ENABLED
            && pursuit
            && game_state == GameState::Run
            && self.wall_striker.is_none()
        {
            if let Some(sid) =
                elect_wall_striker(&world, &self.last_wall_ids, self.double_touch_robot)
            {
                // Straight-forward reflex clear: kick through the incoming ball
                // toward the opponent backline at the striker's own lateral lane.
                let sy = world.own_player(sid).map(|p| p.position.y).unwrap_or(0.0);
                let target = Vector2::new(world.field_length() / 2.0, sy);
                self.planner.clear_plan();
                self.driver.set_waypoint(
                    planner::Waypoint::Handle {
                        action: BallAction::Strike {
                            target,
                            acquire_first: false,
                        },
                        rescue: false,
                    },
                    sid,
                );
                self.wall_striker = Some(sid);
            }
        }
        if let Some(sid) = self.wall_striker {
            // A plan slot: Formation must not position it, and its own body holds
            // its wall slot (Shadow slot-hold) so no backfill is pulled in.
            reserved.push(sid);
            debug::string("wall_striker", &sid.as_u32().to_string());
        } else {
            debug::string("wall_striker", "none");
        }

        // ── Pursuit: hand Formation a capture role so its matching elects the
        //    ball-winner (weighed against defensive duty). While a wall strike is
        //    active the open capture is suppressed — the striker owns the ball and
        //    nobody else may chase it into the wall. ───────────────────────────────
        let capture = (pursuit && self.wall_striker.is_none()).then(|| {
            let ball = world.ball_position().unwrap_or_default();
            let pos = world
                .predict_ball_position(config::CAPTURE_LEAD_TAU)
                .unwrap_or(ball);
            // Only the double-touch robot is barred from capturing (rule compliance);
            // there is no failure-based exclusion — a stuck robot re-engages.
            let ineligible = self.double_touch_robot.into_iter().collect();
            formation::CaptureRole {
                pos,
                importance: config::CAPTURE_IMPORTANCE,
                ineligible,
                // During our set piece the capture role is reserved for the
                // pinned taker — matching may not hand the ball to anyone else.
                pinned: if take_window {
                    self.restart_kicker
                } else {
                    None
                },
            }
        });

        let plan_context = self.driver.plan_context();

        // ── Formation: one cost-aware matching over all field robots (the capturer
        //    is one of the roles in pursuit). ─────────────────────────────────────
        let fout = self
            .formation
            .update(&world, &reserved, &plan_context, capture.as_ref(), now);

        // First live tick of our restart with no prep pin (direct jump to a
        // live set piece): latch the taker. Prefer the framework's identity
        // latch (a robot already mid-take, e.g. after a strategy restart) over
        // the free election. Never latch once the kick has been released — the
        // post-kick collector is not a kicker.
        if take_window && self.restart_kicker.is_none() && world.double_touch_barred().is_none() {
            self.restart_kicker = world.freekick_kicker().or(fout.capturer);
        }

        // ── Pursuit: build the capture waypoint for Formation's chosen capturer.
        //    Skipped while a wall strike runs — the striker's waypoint was set
        //    directly above and must not be torn down by the capture replan
        //    (`fout.capturer` is None then, which would read as a mismatch). ──────
        if pursuit && self.wall_striker.is_none() {
            let needs_replan = self.planner.current_plan().is_none()
                || matches!(
                    self.driver.status(),
                    WaypointStatus::Succeeded | WaypointStatus::Failed(_)
                )
                || possession_changed
                || contest_changed
                || game_state_changed
                || fout.capturer != self.driver.active_robot_id();
            if needs_replan {
                let plan = fout
                    .capturer
                    .and_then(|c| self.planner.replan(&world, &possession, Some(c), &inputs));
                match plan {
                    Some(plan) => self
                        .driver
                        .set_waypoint(plan.waypoints[0].clone(), plan.active_robot),
                    // Fix C: only tear down when there is genuinely no one to
                    // pursue. If a capturer is elected but the plan momentarily
                    // can't form (e.g. a one-frame ball-detection gap), keep the
                    // existing waypoint so the pursuer stays commanded instead of
                    // dropping to "unassigned" for a frame and re-acquiring next
                    // tick — a source of capturing↔unassigned churn.
                    None if fout.capturer.is_none() => {
                        self.planner.clear_plan();
                        self.driver.clear();
                    }
                    None => {}
                }
            }
        }
        // Drive the active pursuit robot — the elected capturer or the wall
        // striker. On a Failed status the next tick re-derives a fresh plan for
        // the same most-eligible robot; no per-robot failure memory.
        if pursuit {
            self.driver.update(&world, ctx);
        }

        // Compliance: name the pinned taker as the kicker during our restarts so
        // the executor exempts it and positions everyone else. Gated on the pin:
        // once the kick is released the pin drops, so the post-kick collector
        // never inherits the label (or the executor's keep-out exemptions) while
        // the GC still reports the restart state.
        if let Some(active_id) = self.driver.active_robot_id() {
            if self.restart_kicker == Some(active_id) {
                if let Some(role) = our_kicker_role(game_state, us_operating) {
                    if let Some(p) = ctx.player(active_id) {
                        p.set_role(role);
                    }
                }
            }
        }

        // Wall-striker candidate pool for next tick: the robots serving Shadow
        // roles now. (The active striker is reserved and absent from commands —
        // irrelevant, since election only runs when no striker is active.)
        self.last_wall_ids = fout
            .commands
            .iter()
            .filter(|c| c.role == "shadow")
            .map(|c| c.id)
            .collect();

        // ── Apply Formation positioning (non-capturing field robots) ─────
        for cmd in &fout.commands {
            if let Some(p) = ctx.player(cmd.id) {
                p.go_to(cmd.target).facing(cmd.face);
                p.set_role(cmd.role);
            }
        }

        // ── Goalkeeper ──────────────────────────────────────────────────
        if let Some(kid) = world.our_keeper_id() {
            if let Some(k) = ctx.player(kid) {
                keeper::update(&mut self.keeper, &world, k);
            }
        }

        self.draw_debug(&world, &possession, &fout.commands);
    }
}

impl ConcertoStrategy {
    /// Pose every field robot (keeper included) into the static logo triangle.
    /// Slots are team-relative; the executor transforms and — for `formation`-role
    /// robots — permits slow repositioning while play is stopped.
    fn run_logo(&self, world: &World, ctx: &mut TeamContext) {
        let face = world.opp_goal_center();
        for p in world.own_players() {
            // ID-based slots: missing IDs leave their spot empty; IDs outside the
            // triangle (>5) are left where they are.
            let Some(slot) = logo::slot_for(p.id) else {
                continue;
            };
            debug::cross(&format!("logo_{}", p.id.as_u32()), slot);
            if let Some(h) = ctx.player(p.id) {
                h.go_to(slot).facing(face);
                h.set_role("formation");
            }
        }
    }

    /// Designate the set-piece kicker, send it to the ball, and name it so the
    /// executor exempts it from restart clamping. The choice is latched in
    /// `restart_kicker`: re-picking the nearest robot every frame flickers
    /// between near-equidistant robots and can disagree with the live election
    /// at the prep→live transition. Nearest-to-ball only picks the initial pin.
    fn designate_prep_kicker(
        &mut self,
        world: &World,
        ctx: &mut TeamContext,
        game_state: GameState,
    ) -> Option<PlayerId> {
        let ball_pos = world.ball_position()?;
        let keeper_id = world.our_keeper_id();
        let id = match self
            .restart_kicker
            .filter(|id| world.own_players().iter().any(|p| p.id == *id))
        {
            Some(id) => id,
            None => {
                world
                    .own_players()
                    .iter()
                    .filter(|p| Some(p.id) != keeper_id)
                    .min_by(|a, b| {
                        let da = (a.position - ball_pos).norm();
                        let db = (b.position - ball_pos).norm();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })?
                    .id
            }
        };
        self.restart_kicker = Some(id);
        let role = match game_state {
            GameState::PrepareKickoff => "kickoff_kicker",
            GameState::FreeKick => "free_kick_kicker",
            _ => "penalty_kicker",
        };
        // During a prepare/setup state the kicker must NOT touch the ball — it may
        // only approach. Going straight to `ball_pos` drives the robot into the
        // ball and dribbles it off-center (a kickoff/penalty rule violation).
        // Stand off behind the ball along the attacking axis so we're poised to
        // pounce the instant the restart goes live without making contact.
        let to_goal = world.opp_goal_center() - ball_pos;
        let attack_dir = to_goal
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector2::new(1.0, 0.0));
        let standoff = ball_pos - attack_dir * PREP_KICKER_STANDOFF;
        if let Some(p) = ctx.player(id) {
            p.go_to(standoff).facing(world.opp_goal_center());
            p.set_role(role);
        }
        Some(id)
    }

    fn draw_debug(
        &self,
        world: &World,
        possession: &Possession,
        commands: &[formation::FormationCommand],
    ) {
        if let Some(ball) = world.ball_position() {
            debug::cross("ball", ball);
        }
        let poss_str = match possession {
            Possession::We(id) => format!("We({})", id.as_u32()),
            Possession::Opp(id) => format!("Opp({})", id.as_u32()),
            Possession::Loose => "Loose".to_string(),
            Possession::Contested => "Contested".to_string(),
        };
        debug::string("possession", &poss_str);

        // Contest: ring the contested ball and draw the squeeze axis to the
        // principal presser, so the deadlock-break behaviour is visible.
        if let (Some(contest), Some(ball)) = (world.ball_contest(), world.ball_position()) {
            debug::string(
                "contest",
                &format!("ours={} opp={}", contest.ours.len(), contest.opp.len()),
            );
            debug::circle_stroke("contest_ball", ball, 250.0, DebugColor::Red);
            if let Some(p) = world
                .principal_presser()
                .and_then(|id| world.opp_player(id))
            {
                debug::line_colored("contest_axis", p.position, ball, DebugColor::Red);
            }
        }

        if let Some(active) = self.driver.active_robot_id() {
            if let Some(p) = world.own_player(active) {
                debug::circle("active_robot", p.position, 200.0);
            }
        }
        for cmd in commands {
            debug::target_colored(
                &format!("formation_{}", cmd.id.as_u32()),
                cmd.id,
                cmd.target,
                DebugColor::Blue,
            );
        }

        self.draw_plan();
    }

    /// Emit the current plan as a structured plan primitive for the UI's plan
    /// panel. No-op when there is no active plan.
    fn draw_plan(&self) {
        let Some(plan) = self.planner.current_plan() else {
            return;
        };

        let steps: Vec<debug::PlanStep> = plan
            .waypoints
            .iter()
            .enumerate()
            // v1 plans are single-waypoint; the head waypoint is the active step.
            .map(|(i, wp)| plan_step(wp, i == 0))
            .collect();

        debug::plan("plan", Some(plan.active_robot.as_u32()), steps);
    }
}

/// Build a UI plan step from a waypoint.
fn plan_step(wp: &planner::Waypoint, active: bool) -> debug::PlanStep {
    use dies_strategy_api::BallAction;
    use planner::Waypoint;
    let (kind, detail) = match wp {
        Waypoint::Steal { from } => ("Steal", format!("from p{}", from.as_u32())),
        Waypoint::Handle { action, rescue } => match action {
            BallAction::Shoot { target } => {
                ("Shoot", format!("→ ({:.0}, {:.0})", target.x, target.y))
            }
            BallAction::Strike { target, .. } => {
                ("Strike", format!("→ ({:.0}, {:.0})", target.x, target.y))
            }
            BallAction::Carry { to, .. } => ("Carry", format!("→ ({:.0}, {:.0})", to.x, to.y)),
            BallAction::Hold { .. } => (
                "Acquire",
                if *rescue { "rescue off line" } else { "ball" }.to_string(),
            ),
        },
        Waypoint::Pass {
            passer,
            receiver,
            target_area,
        } => (
            "Pass",
            format!(
                "p{}→p{} @ ({:.0}, {:.0})",
                passer.as_u32(),
                receiver.as_u32(),
                target_area.x,
                target_area.y
            ),
        ),
    };
    debug::PlanStep {
        kind: kind.to_string(),
        label: kind.to_string(),
        detail: Some(detail),
        active,
    }
}

/// Role name to apply to the active robot when it's the kicker for our restart.
fn our_kicker_role(game_state: GameState, us_operating: bool) -> Option<&'static str> {
    if !us_operating {
        return None;
    }
    match game_state {
        GameState::Kickoff | GameState::PrepareKickoff => Some("kickoff_kicker"),
        GameState::FreeKick => Some("free_kick_kicker"),
        GameState::Penalty | GameState::PreparePenalty | GameState::PenaltyRun => {
            Some("penalty_kicker")
        }
        _ => None,
    }
}

/// Elect the wall robot to reflex-strike an incoming ball: the ball must be
/// free-rolling toward our side above [`config::WALL_STRIKE_MIN_SPEED`], and the
/// striker is the wall robot nearest the ball's line of travel — reachable with
/// a step ([`config::WALL_STRIKE_REACH`]), arriving within
/// [`config::WALL_STRIKE_MAX_TTC`]. The wall robot in the path is the low-risk
/// choice by construction: it barely moves, stays on the ball line (the line
/// that matters), and its emptied slot is held by its own body.
fn elect_wall_striker(
    world: &World,
    wall_ids: &[PlayerId],
    barred: Option<PlayerId>,
) -> Option<PlayerId> {
    let ball = world.ball_position()?;
    let vel = world.ball_velocity()?;
    // Incoming: fast enough to be a delivery, and travelling toward our side.
    if vel.norm() < config::WALL_STRIKE_MIN_SPEED || vel.x >= 0.0 {
        return None;
    }
    let mut best: Option<(PlayerId, f64)> = None;
    for id in wall_ids {
        if Some(*id) == barred {
            continue;
        }
        let Some(p) = world.own_player(*id) else {
            continue;
        };
        let Some((t, miss)) = geometry::ball_closest_approach(ball, vel, p.position) else {
            continue;
        };
        if t <= 0.0 || t > config::WALL_STRIKE_MAX_TTC || miss > config::WALL_STRIKE_REACH {
            continue;
        }
        if best.map(|(_, m)| miss < m).unwrap_or(true) {
            best = Some((*id, miss));
        }
    }
    best.map(|(id, _)| id)
}

/// Whether an armed wall strike is still worth finishing. Looser than the entry
/// gate (hysteresis): while the ball still rolls, its approach must remain
/// plausible for this striker; once it has (nearly) died, the striker finishes
/// the poke-clear iff the ball rests within a couple of steps — never leaving a
/// stopped ball sitting in front of our goal for an opponent to run onto.
fn wall_strike_valid(world: &World, striker: PlayerId) -> bool {
    let (Some(ball), Some(vel), Some(p)) = (
        world.ball_position(),
        world.ball_velocity(),
        world.own_player(striker),
    ) else {
        return false;
    };
    if vel.norm() > config::WALL_STRIKE_EXIT_SPEED {
        // Still rolling: kicked/deflected away (receding or wide) ends the
        // episode; margins are grown so normal approach jitter doesn't flicker.
        let Some((t, miss)) = geometry::ball_closest_approach(ball, vel, p.position) else {
            return false;
        };
        vel.x < 0.0
            && t > -0.15
            && t < config::WALL_STRIKE_MAX_TTC * 1.5
            && miss < config::WALL_STRIKE_REACH * 1.4
    } else {
        // Ball has died nearby: finish striking it clear.
        (ball - p.position).norm() < config::WALL_STRIKE_REACH * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{BallState, WorldSnapshot};

    fn world_with_ball(
        own: Vec<PlayerState>,
        ball_pos: Vector2,
        ball_vel: Vector2,
    ) -> World {
        World::new(WorldSnapshot {
            timestamp: 0.0,
            dt: 0.016,
            field_geom: Some(FieldGeometry::default()),
            ball: Some(BallState {
                position: ball_pos,
                velocity: ball_vel,
                detected: true,
            }),
            own_players: own,
            opp_players: vec![],
            game_state: GameState::Run,
            us_operating: true,
            pre_stage: false,
            our_keeper_id: Some(PlayerId::new(1)),
            freekick_kicker: None,
            possession: Possession::Loose,
            possession_stale: false,
            ball_contest: None,
            double_touch_barred: None,
        })
    }

    fn player(id: u32, x: f64, y: f64) -> PlayerState {
        PlayerState::new(
            PlayerId::new(id),
            Vector2::new(x, y),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        )
    }

    #[test]
    fn wall_striker_is_the_robot_in_the_ball_path() {
        // Wall at x=-3000; ball rolling straight at the +y wing (id 3).
        let own = vec![
            player(1, -4300.0, 0.0), // keeper
            player(2, -3000.0, 0.0),
            player(3, -3000.0, 400.0),
            player(4, -3000.0, -400.0),
        ];
        let wall = vec![PlayerId::new(2), PlayerId::new(3), PlayerId::new(4)];
        let w = world_with_ball(own, Vector2::new(-1000.0, 400.0), Vector2::new(-1800.0, 0.0));
        assert_eq!(elect_wall_striker(&w, &wall, None), Some(PlayerId::new(3)));
    }

    #[test]
    fn no_striker_for_slow_wide_or_outgoing_balls() {
        let own = vec![
            player(1, -4300.0, 0.0),
            player(2, -3000.0, 0.0),
            player(3, -3000.0, 400.0),
        ];
        let wall = vec![PlayerId::new(2), PlayerId::new(3)];
        // Too slow.
        let w = world_with_ball(
            own.clone(),
            Vector2::new(-1000.0, 0.0),
            Vector2::new(-300.0, 0.0),
        );
        assert_eq!(elect_wall_striker(&w, &wall, None), None);
        // Moving away from our side.
        let w = world_with_ball(
            own.clone(),
            Vector2::new(-1000.0, 0.0),
            Vector2::new(1500.0, 0.0),
        );
        assert_eq!(elect_wall_striker(&w, &wall, None), None);
        // Fast but passing far wide of every wall robot.
        let w = world_with_ball(
            own.clone(),
            Vector2::new(-1000.0, 2500.0),
            Vector2::new(-1800.0, 0.0),
        );
        assert_eq!(elect_wall_striker(&w, &wall, None), None);
        // Too far out: closest approach beyond the TTC gate.
        let w = world_with_ball(own, Vector2::new(2000.0, 0.0), Vector2::new(-600.0, 0.0));
        assert_eq!(elect_wall_striker(&w, &wall, None), None);
    }

    #[test]
    fn strike_stays_valid_for_a_ball_dying_within_reach() {
        // The rejected-MVP failure: a roller that dies in front of the wall must
        // still be poked clear by the armed striker, not left for an opponent.
        let own = vec![player(1, -4300.0, 0.0), player(2, -3000.0, 0.0)];
        let sid = PlayerId::new(2);
        // Nearly stopped 600mm in front of the striker → finish the clear.
        let w = world_with_ball(
            own.clone(),
            Vector2::new(-2400.0, 0.0),
            Vector2::new(-50.0, 0.0),
        );
        assert!(wall_strike_valid(&w, sid));
        // Stopped far away → episode ends, normal capture resumes.
        let w = world_with_ball(
            own.clone(),
            Vector2::new(-500.0, 0.0),
            Vector2::new(-50.0, 0.0),
        );
        assert!(!wall_strike_valid(&w, sid));
        // Kicked clear (receding fast) → episode ends.
        let w = world_with_ball(own, Vector2::new(-2400.0, 0.0), Vector2::new(3000.0, 0.0));
        assert!(!wall_strike_valid(&w, sid));
    }
}
