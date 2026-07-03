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
pub mod penalty;
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

/// Own-half clamp (robot-center x, mm) for non-kicker staging targets while a
/// kickoff is being prepared. Deeper than the executor's `comply()` clamp
/// (-190mm) so that clamp is a no-op once the real `PrepareKickoff` arrives —
/// we're already fully staged during the `pre_stage` window, where `comply()`
/// still keys off the real Stop state and enforces nothing about the halfway line.
const KICKOFF_STAGE_X: f64 = -300.0;

/// Extra standoff (mm) beyond the center-circle radius for kickoff staging
/// targets. `comply()` pushes non-kickers out to radius + 500 in the real
/// PrepareKickoff/Kickoff states; stage slightly deeper so that's a no-op too.
const KICKOFF_CIRCLE_MARGIN: f64 = 600.0;

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
    /// Latched penalty-taker state (aim point), reset outside the penalty family.
    penalty: penalty::PenaltyState,
    /// Wall robot currently stepping out to clear a slow/settled ball in the
    /// apron (see `config::WALL_CLEAR_ENABLED`). While set, it is a reserved plan
    /// slot (its own body holds its wall slot via the Shadow slot-hold). The open
    /// capture role is suppressed by the apron zone itself, not by this field.
    wall_clearer: Option<PlayerId>,
    /// Robots serving Shadow roles last tick (from Formation's commands) — the
    /// wall-clearer candidate pool. One tick stale by construction, which is fine
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
            penalty: penalty::PenaltyState::default(),
            wall_clearer: None,
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
        // Per-role formation switches: turning one off removes the role from the
        // matching entirely (the body is freed for other roles, never parked on a
        // zero-importance slot). The capturer has no switch of its own — the
        // `defense_only` toggle already suppresses it (it kills `pursuit`, so no
        // CaptureRole is ever handed to Formation).
        vec![
            ParamSpec::bool("defense_only", "Defense only (suppress offense)", false),
            ParamSpec::bool(
                "warmup",
                "Warmup: pose in the logo (triangle) formation",
                false,
            ),
            ParamSpec::bool("role_wall", "Role: wall (shadow wings + anchor)", true),
            ParamSpec::bool("role_mark", "Role: man-marking", true),
            ParamSpec::bool("role_support", "Role: wide supporters", true),
            ParamSpec::bool("role_striker", "Role: striker / box-runner", true),
            ParamSpec::bool("role_pivot", "Role: recycle pivot", true),
            ParamSpec::bool("role_balance", "Role: rest defender", true),
            ParamSpec::bool("role_outlet", "Role: counter outlet", true),
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
        let penalty_family = matches!(
            game_state,
            GameState::PreparePenalty | GameState::Penalty | GameState::PenaltyRun
        );
        let take_window = us_operating
            && (penalty_family
                || matches!(
                    game_state,
                    GameState::PrepareKickoff | GameState::Kickoff | GameState::FreeKick
                ));
        if !take_window {
            self.restart_kicker = None;
        } else if let Some(pin) = self.restart_kicker {
            // Release when: the taker released the ball into play (pin→bar
            // handoff — the bar then keeps it off the ball), a different own
            // robot touched first (interference/steal of the take), or the
            // pinned robot left the field (sidelined/carded → re-latch).
            // EXCEPT during a penalty: double touch does not apply there
            // (rules §8.2 covers only kick-off/free kick), the taker may
            // legally re-touch a rebound, and dropping the pin would strip its
            // `penalty_kicker` exemption mid-kick — the executor would park it.
            let taken = !penalty_family && world.double_touch_barred() == Some(pin);
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

        // ── Penalty kicks (rules §5.3.5) ────────────────────────────────
        // The executor enforces all penalty positioning — every robot except
        // the taker (ours) / the keeper (theirs) is parked at the field end and
        // clamped ≥1m behind the ball — so both branches drive only their one
        // privileged robot and return; Formation/Planner output would be dead
        // weight here. Covers the *predicted* `PreparePenalty` during a
        // stoppage too (`pre_stage`), which stages taker/keeper early.
        if !penalty_family {
            self.penalty.reset();
        } else {
            self.planner.clear_plan();
            self.driver.clear();
            if us_operating {
                // Ours: pin a taker (nearest non-keeper on first designation)
                // and stage it / take the shot.
                let anchor = world
                    .ball_position()
                    .unwrap_or_else(|| penalty::opp_penalty_mark(&world));
                if let Some(id) = self.elect_kicker(&world, anchor) {
                    penalty::update_taker(&mut self.penalty, &world, ctx, id, game_state);
                }
            } else if let Some(k) = world.our_keeper_id().and_then(|kid| ctx.player(kid)) {
                // Theirs: the keeper holds the goal line until the kick is
                // live (keeper.rs penalty gate), then guards/intercepts.
                keeper::update(&mut self.keeper, &world, k);
            }
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
            // Penalty states never reach here (dedicated branch above).
            GameState::Kickoff | GameState::FreeKick => us_operating,
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
        // otherwise clamp/sideline our kicker: PrepareKickoff, and a *predicted*
        // free kick during a stoppage (`pre_stage`): there is no prepare-variant
        // for free kicks, so we line the taker up here instead of running live
        // offense. (Penalty prep lives in the dedicated penalty branch above.)
        if us_operating
            && (game_state == GameState::PrepareKickoff
                || (pre_stage && game_state == GameState::FreeKick))
        {
            if let Some(id) = self.designate_prep_kicker(&world, ctx, game_state) {
                reserved.push(id);
            }
        }

        // ── Wall clear: a slow/settled ball in the apron (in front of our wall,
        //    outside the keeper's box) belongs to the wall alone. `wall_owns`
        //    unconditionally suppresses the open capture (below) so no outside
        //    capturer stern-chases the ball into the corridor. A *fast* ball is
        //    blocked by the standing wall — nobody steps out. A *slow* one is
        //    hoofed clear by the nearest waller, but only when we win the race to
        //    it. Exit first (ball left the apron / episode resolved / geometry
        //    lapsed / another mode owns the driver), then arm. ───────────────────
        let wall_owns = config::WALL_CLEAR_ENABLED
            && pursuit
            && game_state == GameState::Run
            && wall_owns_ball(&world);
        if let Some(sid) = self.wall_clearer {
            if !pursuit {
                // Offense or a stoppage owns the driver now (it was replanned or
                // cleared above); just drop the clearer designation.
                self.wall_clearer = None;
            } else if !wall_owns
                || matches!(
                    self.driver.status(),
                    WaypointStatus::Succeeded | WaypointStatus::Failed(_)
                )
                || !wall_clear_valid(&world, sid)
            {
                // Ball left the apron, a Failed whiff, or geometry lapsed: drop and
                // clear. If the ball is still clearable the entry below re-arms next
                // tick (always re-engage, never rotate).
                self.wall_clearer = None;
                self.driver.clear();
            }
        }
        if wall_owns && self.wall_clearer.is_none() {
            if let Some(sid) =
                elect_wall_clearer(&world, &self.last_wall_ids, self.double_touch_robot)
            {
                // Straight-forward hoof: kick the settled ball toward the opponent
                // backline at the clearer's own lateral lane.
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
                self.wall_clearer = Some(sid);
            }
        }
        if let Some(sid) = self.wall_clearer {
            // A plan slot: Formation must not position it, and its own body holds
            // its wall slot (Shadow slot-hold) so no backfill is pulled in.
            reserved.push(sid);
            debug::string("wall_clearer", &sid.as_u32().to_string());
        } else {
            debug::string("wall_clearer", "none");
        }

        // ── Pursuit: hand Formation a capture role so its matching elects the
        //    ball-winner (weighed against defensive duty). While the ball is in the
        //    wall apron (`wall_owns`) the open capture is suppressed — the wall owns
        //    the ball and nobody else may chase it into the corridor, whether or not
        //    a clearer was elected. A ball inside our own box is likewise never
        //    captured: a field robot can't legally take it (multiple-defenders
        //    penalty) and the pursuer just camps at the keepout line on the keeper's
        //    clear approach. ──────────────────────────────────────────────────────
        let ball_in_our_box = world
            .ball_position()
            .map(|b| world.own_penalty_area().contains(b))
            .unwrap_or(false);
        let capture = (pursuit && !wall_owns && !ball_in_our_box).then(|| {
            let ball = world.ball_position().unwrap_or_default();
            let pos = world
                .predict_ball_position(config::CAPTURE_LEAD_TAU)
                .unwrap_or(ball);
            // The double-touch robot is barred from capturing (rule compliance),
            // as is any robot that can't kick or dribble (handicaps) — unless
            // every field robot is handicapped, in which case a crippled
            // shepherd still beats an unchased ball. There is no failure-based
            // exclusion — a stuck robot re-engages.
            let mut ineligible: Vec<PlayerId> = self.double_touch_robot.into_iter().collect();
            let keeper_id = world.our_keeper_id();
            let any_able = world
                .own_players()
                .iter()
                .any(|p| Some(p.id) != keeper_id && !ball_duty_barred(p));
            if any_able {
                ineligible.extend(
                    world
                        .own_players()
                        .iter()
                        .filter(|p| ball_duty_barred(p))
                        .map(|p| p.id),
                );
            }
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
        self.formation.set_toggles(formation::RoleToggles {
            wall: ctx.param_bool("role_wall"),
            mark: ctx.param_bool("role_mark"),
            support: ctx.param_bool("role_support"),
            striker: ctx.param_bool("role_striker"),
            pivot: ctx.param_bool("role_pivot"),
            balance: ctx.param_bool("role_balance"),
            outlet: ctx.param_bool("role_outlet"),
        });
        // One frame stale (keeper::update runs after the formation below) —
        // harmless: the wall takes several frames to part anyway.
        let keeper_clearing = self.keeper.is_clearing();
        let fout = self.formation.update(
            &world,
            &reserved,
            &plan_context,
            capture.as_ref(),
            now,
            keeper_clearing,
        );

        // First live tick of our restart with no prep pin (direct jump to a
        // live set piece): latch the taker. Prefer the framework's identity
        // latch (a robot already mid-take, e.g. after a strategy restart) over
        // the free election. Never latch once the kick has been released — the
        // post-kick collector is not a kicker.
        if take_window && self.restart_kicker.is_none() && world.double_touch_barred().is_none() {
            // The framework latch can name a handicapped robot (it only tracks
            // who is nearest/mid-take) — a robot that can't kick must not be
            // pinned as taker; fall through to the election, which respects
            // capture ineligibility.
            self.restart_kicker = world
                .freekick_kicker()
                .filter(|id| {
                    world
                        .own_player(*id)
                        .map(|p| !ball_duty_barred(p))
                        .unwrap_or(true)
                })
                .or(fout.capturer);
        }

        // ── Pursuit: build the capture waypoint for Formation's chosen capturer.
        //    Skipped while a wall clearer runs — the clearer's waypoint was set
        //    directly above and must not be torn down by the capture replan
        //    (`fout.capturer` is None then, which would read as a mismatch). In
        //    apron hold-mode (no clearer) this runs, sees no capturer, and tears
        //    down any stale pursuit so nobody drives into the wall. ───────────────
        if pursuit && self.wall_clearer.is_none() {
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
        // clearer. On a Failed status the next tick re-derives a fresh plan for
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

        // Wall-clearer candidate pool for next tick: the robots serving Shadow
        // roles now. (The active clearer is reserved and absent from commands —
        // irrelevant, since election only runs when no clearer is active.)
        self.last_wall_ids = fout
            .commands
            .iter()
            .filter(|c| c.role == "shadow")
            .map(|c| c.id)
            .collect();

        // ── Apply Formation positioning (non-capturing field robots) ─────
        // Kickoff staging: while a kickoff is being prepared — including the
        // predicted `pre_stage` window where the GC still shows Stop — every
        // non-kicker stages on our half, clear of the center circle, so nobody
        // scrambles back when the real PrepareKickoff clamps arrive. The kicker
        // is never in these commands (reserved in prep, capturer when live).
        let kickoff_staging = matches!(game_state, GameState::PrepareKickoff | GameState::Kickoff);
        let circle_keepout = world
            .field()
            .map(|f| f.center_circle_radius)
            .unwrap_or(500.0)
            + KICKOFF_CIRCLE_MARGIN;
        for cmd in &fout.commands {
            if let Some(p) = ctx.player(cmd.id) {
                let mut target = cmd.target;
                if kickoff_staging {
                    target.x = target.x.min(KICKOFF_STAGE_X);
                    if target.norm() < circle_keepout {
                        // Slide out of the center circle along -x; with x already
                        // ≤ KICKOFF_STAGE_X this stays on our half.
                        target.x = -(circle_keepout * circle_keepout - target.y * target.y)
                            .max(0.0)
                            .sqrt();
                    }
                }
                // Shadows (the wall) ride a rounded-rect bounded path around the box:
                // keeper-style direct-velocity drive with a no-overshoot envelope,
                // ORCA + planner off. The envelope keeps the pair locked to the curve
                // (and out of the box) so ORCA can't split them at a corner and the
                // rigid base_s keeps their spacing pinned. A waller that's still far
                // outside its band drives in with the planner first (go_to_bounded
                // bypasses it and would cut straight through obstacles), handing off
                // to bounded once inside. During keeper-clearing bounds is None (set
                // by the generator) so the wall vacates on the planner, not the
                // envelope. All other roles use the planned go_to as before.
                match cmd.bounds {
                    Some(bounds) => {
                        let near_band = match bounds {
                            MotionBounds::RoundedRect(z) => {
                                let d = z.center
                                    + Vector2::new(
                                        (p.position().x - z.center.x)
                                            .clamp(-z.half_extents.x, z.half_extents.x),
                                        (p.position().y - z.center.y)
                                            .clamp(-z.half_extents.y, z.half_extents.y),
                                    );
                                (p.position() - d).norm()
                                    <= z.max_offset + config::WALL_HANDOFF_SLACK
                            }
                            _ => true,
                        };
                        if near_band {
                            p.go_to_bounded(target, bounds).facing(cmd.face);
                        } else {
                            p.go_to(target).facing(cmd.face).avoid_robots(false);
                        }
                    }
                    None => {
                        p.go_to(target)
                            .facing(cmd.face)
                            .avoid_robots(cmd.role != "shadow");
                    }
                }
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
    /// Resolve the pinned taker of our set piece: keep the latched pin while
    /// its robot is on the field, otherwise elect the non-keeper nearest to
    /// `anchor` (the ball, or the expected ball spot) and latch it.
    fn elect_kicker(&mut self, world: &World, anchor: Vector2) -> Option<PlayerId> {
        let keeper_id = world.our_keeper_id();
        let id = match self
            .restart_kicker
            .filter(|id| world.own_players().iter().any(|p| p.id == *id))
        {
            Some(id) => id,
            None => {
                // Prefer a robot that can actually take the kick; only if every
                // field robot is handicapped does the nearest get it anyway.
                let nearest = |allow_barred: bool| {
                    world
                        .own_players()
                        .iter()
                        .filter(|p| {
                            Some(p.id) != keeper_id && (allow_barred || !ball_duty_barred(p))
                        })
                        .min_by(|a, b| {
                            let da = (a.position - anchor).norm();
                            let db = (b.position - anchor).norm();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|p| p.id)
                };
                nearest(false).or_else(|| nearest(true))?
            }
        };
        self.restart_kicker = Some(id);
        Some(id)
    }

    fn designate_prep_kicker(
        &mut self,
        world: &World,
        ctx: &mut TeamContext,
        game_state: GameState,
    ) -> Option<PlayerId> {
        let ball_pos = world.ball_position()?;
        let id = self.elect_kicker(world, ball_pos)?;
        let role = match game_state {
            GameState::PrepareKickoff => "kickoff_kicker",
            _ => "free_kick_kicker",
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
        // Penalty states never reach the shared path — the dedicated penalty
        // branch sets `penalty_kicker` itself.
        _ => None,
    }
}

/// The ball is in the wall's "apron": loose in front of our wall, near our own
/// goal, outside the keeper's box. While true, the wall owns the ball — the open
/// capture role is suppressed (nobody stern-chases it into the corridor) and the
/// wall decides between blocking (fast ball) and clearing (slow ball). The wall
/// always re-centers on the ball→goal line via `shadow_arc`, so a radial x-depth
/// plus a lateral bound suffices — no separate angular test is needed.
fn wall_owns_ball(world: &World) -> bool {
    let Some(ball) = world.ball_position() else {
        return false;
    };
    // Inside our own box is the keeper's domain, not the wall's.
    if world.own_penalty_area().contains(ball) {
        return false;
    }
    let front = -world.field_length() / 2.0 + config::WALL_CLEAR_APRON_DEPTH;
    ball.x < front && ball.y.abs() < config::WALL_CLEAR_APRON_HALF_WIDTH
}

/// Time-to-ball for a robot via the momentum-aware `redirect_time` cost.
fn time_to_ball(pos: Vector2, vel: Vector2, ball: Vector2) -> f64 {
    geometry::redirect_time(pos, vel, ball, config::V_MAX, config::A_MAX)
}

/// Closest opponent's time-to-ball, `+inf` if there is no opponent.
fn opp_time_to_ball(world: &World, ball: Vector2) -> f64 {
    world
        .closest_opp_player_to(ball)
        .map(|o| time_to_ball(o.position, o.velocity, ball))
        .unwrap_or(f64::INFINITY)
}

/// A robot missing its kicker or dribbler can't do ball duty (capture, set-piece
/// kicks): it can neither strike nor hold the ball. `NoBreakbeam` doesn't bar
/// anything — the ToF backup covers ball detection.
fn ball_duty_barred(p: &PlayerState) -> bool {
    p.handicaps.contains(&Handicap::NoKicker) || p.handicaps.contains(&Handicap::NoDribbler)
}

/// Elect a wall robot to step out and hoof a slow/settled apron ball clear. Only
/// slow balls ([`config::WALL_CLEAR_MAX_SPEED`]) — a faster ball is a shot/
/// delivery and is blocked by the standing wall, never chased out of formation.
/// The clearer is the waller nearest the ball by [`time_to_ball`], and we only
/// commit when it wins the race to the ball against the closest opponent (within
/// [`config::WALL_CLEAR_MARGIN`]); losing the race would open a wall gap AND hand
/// them a dead ball, so we hold instead.
fn elect_wall_clearer(
    world: &World,
    wall_ids: &[PlayerId],
    barred: Option<PlayerId>,
) -> Option<PlayerId> {
    let ball = world.ball_position()?;
    let vel = world.ball_velocity()?;
    // Too fast to step out for — the standing wall blocks it.
    if vel.norm() > config::WALL_CLEAR_MAX_SPEED {
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
        // The hoof is a straight kick — a robot without a kicker would just
        // nudge the ball around the apron (no dribbler needed, though).
        if p.handicaps.contains(&Handicap::NoKicker) {
            continue;
        }
        let ttb = time_to_ball(p.position, p.velocity, ball);
        if best.map(|(_, b)| ttb < b).unwrap_or(true) {
            best = Some((*id, ttb));
        }
    }
    let (id, waller_ttb) = best?;
    println!(
        "elecing wall clearer {} {} {}",
        waller_ttb,
        opp_time_to_ball(world, ball),
        config::WALL_CLEAR_MARGIN
    );
    // Win the race (harass-aggressive at margin 0) or hold.
    (waller_ttb <= opp_time_to_ball(world, ball) + config::WALL_CLEAR_MARGIN).then_some(id)
}

/// Whether an armed wall clear is still worth finishing. Looser than the entry
/// gate (hysteresis): the ball must still be in the apron; if it has come to our
/// feet (within [`config::WALL_CLEAR_REACH`]) we finish regardless of the race —
/// never leave a dead ball in front of goal — otherwise we keep going only while
/// not decisively out-raced ([`config::WALL_CLEAR_COMMIT`] slack), so a
/// marginally-closer opponent mid-approach doesn't flicker the wall gap open.
fn wall_clear_valid(world: &World, clearer: PlayerId) -> bool {
    if !wall_owns_ball(world) {
        return false;
    }
    let (Some(ball), Some(p)) = (world.ball_position(), world.own_player(clearer)) else {
        return false;
    };
    // Ball already at our feet: finish the clear regardless of the race.
    if (ball - p.position).norm() < config::WALL_CLEAR_REACH {
        return true;
    }
    let waller_ttb = time_to_ball(p.position, p.velocity, ball);
    opp_time_to_ball(world, ball) + config::WALL_CLEAR_COMMIT >= waller_ttb
}

#[cfg(test)]
mod tests {
    use super::*;
    use dies_strategy_protocol::{BallState, WorldSnapshot};

    fn world_with_ball(
        own: Vec<PlayerState>,
        opp: Vec<PlayerState>,
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
            opp_players: opp,
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

    // Standard defensive wall on a default 9000mm field: keeper + three wallers
    // at the standoff line (x≈-2950 for a square-on ball).
    fn wall_own() -> Vec<PlayerState> {
        vec![
            player(1, -4300.0, 0.0), // keeper
            player(2, -2950.0, 0.0),
            player(3, -2950.0, 400.0),
            player(4, -2950.0, -400.0),
        ]
    }
    fn wall_ids() -> Vec<PlayerId> {
        vec![PlayerId::new(2), PlayerId::new(3), PlayerId::new(4)]
    }

    #[test]
    fn clearer_is_the_nearest_waller_to_a_settled_apron_ball() {
        // Slow ball settled in the apron nearer the +y waller (id 3), nobody
        // contesting → the nearest waller by time-to-ball is elected.
        let w = world_with_ball(
            wall_own(),
            vec![],
            Vector2::new(-2400.0, 300.0),
            Vector2::new(-40.0, 0.0),
        );
        assert_eq!(
            elect_wall_clearer(&w, &wall_ids(), None),
            Some(PlayerId::new(3))
        );
    }

    #[test]
    fn no_clearer_for_a_fast_ball_block_dont_step_out() {
        // A hard shot into the wall is blocked by the standing wall, never chased
        // out of formation — no clearer is elected regardless of geometry.
        let w = world_with_ball(
            wall_own(),
            vec![],
            Vector2::new(-1000.0, 300.0),
            Vector2::new(-1800.0, 0.0),
        );
        assert_eq!(elect_wall_clearer(&w, &wall_ids(), None), None);
    }

    #[test]
    fn no_clearer_when_an_opponent_wins_the_race() {
        // Slow apron ball, but an opponent is clearly closer → we hold the wall
        // rather than step out and lose the race (opening a gap for nothing).
        let w = world_with_ball(
            wall_own(),
            vec![player(9, -2100.0, 0.0)], // ~300mm from the ball
            Vector2::new(-2400.0, 0.0),    // ~550mm from the nearest waller
            Vector2::new(-40.0, 0.0),
        );
        assert_eq!(elect_wall_clearer(&w, &wall_ids(), None), None);
        // Same ball, opponent now far up-field → we win and commit.
        let w = world_with_ball(
            wall_own(),
            vec![player(9, 500.0, 0.0)],
            Vector2::new(-2400.0, 0.0),
            Vector2::new(-40.0, 0.0),
        );
        assert!(elect_wall_clearer(&w, &wall_ids(), None).is_some());
    }

    #[test]
    fn barred_robot_is_not_elected_as_clearer() {
        // The double-touch robot may not take the ball even if it is nearest.
        let w = world_with_ball(
            wall_own(),
            vec![],
            Vector2::new(-2400.0, 400.0), // nearest to id 3
            Vector2::new(-40.0, 0.0),
        );
        let got = elect_wall_clearer(&w, &wall_ids(), Some(PlayerId::new(3)));
        assert!(got.is_some() && got != Some(PlayerId::new(3)));
    }

    #[test]
    fn no_kicker_waller_is_not_elected_as_clearer() {
        // The nearest waller (id 3) has no kicker — it can't hoof, so another
        // waller takes the clear. A missing dribbler doesn't bar the hoof.
        let own = vec![
            player(1, -4300.0, 0.0), // keeper
            player(2, -2950.0, 0.0),
            player(3, -2950.0, 400.0).with_handicaps([Handicap::NoKicker].into()),
            player(4, -2950.0, -400.0).with_handicaps([Handicap::NoDribbler].into()),
        ];
        let w = world_with_ball(
            own,
            vec![],
            Vector2::new(-2400.0, 400.0), // nearest to id 3
            Vector2::new(-40.0, 0.0),
        );
        let got = elect_wall_clearer(&w, &wall_ids(), None);
        assert!(got.is_some() && got != Some(PlayerId::new(3)));
    }

    #[test]
    fn wall_owns_ball_only_in_the_apron() {
        let stopped = Vector2::new(0.0, 0.0);
        // In the apron: in front of the wall, near goal, outside the box.
        let w = world_with_ball(wall_own(), vec![], Vector2::new(-2400.0, 0.0), stopped);
        assert!(wall_owns_ball(&w));
        // Inside our own box → keeper's domain, not the wall's.
        let w = world_with_ball(wall_own(), vec![], Vector2::new(-4000.0, 0.0), stopped);
        assert!(!wall_owns_ball(&w));
        // Too far up-field → a normal loose ball for the capturer.
        let w = world_with_ball(wall_own(), vec![], Vector2::new(-1000.0, 0.0), stopped);
        assert!(!wall_owns_ball(&w));
        // Wide near our backline corner → out of the wall's zone.
        let w = world_with_ball(wall_own(), vec![], Vector2::new(-2400.0, 1800.0), stopped);
        assert!(!wall_owns_ball(&w));
    }

    #[test]
    fn clear_stays_valid_at_our_feet_but_drops_outside_the_apron() {
        let sid = PlayerId::new(2); // waller at (-2950, 0)
                                    // Dead ball at the clearer's feet in the apron → finish regardless of an
                                    // opponent standing right there.
        let w = world_with_ball(
            wall_own(),
            vec![player(9, -2300.0, 0.0)],
            Vector2::new(-2700.0, 0.0), // ~250mm from the clearer
            Vector2::new(-30.0, 0.0),
        );
        assert!(wall_clear_valid(&w, sid));
        // Ball kicked clear out of the apron → episode ends, capture resumes.
        let w = world_with_ball(
            wall_own(),
            vec![],
            Vector2::new(-1000.0, 0.0),
            Vector2::new(2000.0, 0.0),
        );
        assert!(!wall_clear_valid(&w, sid));
    }
}
