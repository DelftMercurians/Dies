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
    /// Ball position when we gained possession (the dribble contact point), for the
    /// excessive-dribbling carry cap. Cleared whenever we don't hold the ball.
    dribble_origin: Option<Vector2>,
    /// World time we gained possession, for the transient fast-break window. Cleared
    /// whenever we don't hold the ball, so it re-stamps on the next turnover.
    gained_ball_t: Option<f64>,
    /// Goalkeeper Guard/Clear state machine.
    keeper: keeper::KeeperState,
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
            dribble_origin: None,
            gained_ball_t: None,
            keeper: keeper::KeeperState::new(),
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
        let now = world.timestamp();
        let defense_only = ctx.param_bool("defense_only");

        // ── Double-touch tracking ───────────────────────────────────────
        if let Some(kicker) = world.freekick_kicker() {
            self.double_touch_robot = Some(kicker);
        } else if matches!(
            game_state,
            GameState::Halt | GameState::Stop | GameState::Timeout | GameState::Run
        ) {
            // Reset on stoppages, and on Run once the framework clears the kicker
            // (another robot has touched the ball).
            self.double_touch_robot = None;
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

        let can_act = !defense_only && ball_present && world.is_ball_in_play() && we_may_act;
        // A pass is atomic: possession flits We→Loose→We while the ball is in flight,
        // so we stay in the offense path (the coordinator owns the pass) rather than
        // treating the transient loose ball as something to chase.
        let passing = self.driver.is_passing();
        // Offense — we hold the ball (or a pass is in flight): the Planner owns the
        // active robot(s). Pursuit — a ball we don't hold but may contest: Formation's
        // matching picks the capturer, weighing it against every defensive duty.
        let offense = can_act && (passing || matches!(possession, Possession::We(_)));
        let pursuit = can_act && !offense;

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
                    Some(plan) => {
                        self.driver
                            .set_waypoint(plan.waypoints[0].clone(), plan.active_robot, now)
                    }
                    None => self.driver.clear(),
                }
            }
            if let Some(active_id) = self.driver.active_robot_id() {
                reserved.extend(self.driver.plan_slots());
                let status = self.driver.update(&world, ctx);
                if let WaypointStatus::Failed(reason) = status {
                    self.planner.record_failure(active_id, reason, now);
                }
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
        // the brief Penalty state before PenaltyRun.
        if us_operating
            && matches!(
                game_state,
                GameState::PrepareKickoff | GameState::PreparePenalty | GameState::Penalty
            )
        {
            if let Some(id) = self.designate_prep_kicker(&world, ctx, game_state) {
                reserved.push(id);
            }
        }

        // ── Pursuit: hand Formation a capture role so its matching elects the
        //    ball-winner (weighed against defensive duty). ───────────────────────
        let capture = pursuit.then(|| {
            let ball = world.ball_position().unwrap_or_default();
            let pos = world
                .predict_ball_position(config::CAPTURE_LEAD_TAU)
                .unwrap_or(ball);
            let mut ineligible = self.planner.capture_ineligible(now);
            ineligible.extend(self.double_touch_robot);
            formation::CaptureRole {
                pos,
                importance: config::CAPTURE_IMPORTANCE,
                ineligible,
            }
        });

        let plan_context = self.driver.plan_context();

        // ── Formation: one cost-aware matching over all field robots (the capturer
        //    is one of the roles in pursuit). ─────────────────────────────────────
        let fout = self
            .formation
            .update(&world, &reserved, &plan_context, capture.as_ref(), now);

        // ── Pursuit: build the capture waypoint for Formation's chosen capturer ──
        if pursuit {
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
                    Some(plan) => {
                        self.driver
                            .set_waypoint(plan.waypoints[0].clone(), plan.active_robot, now)
                    }
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
            if let Some(active_id) = self.driver.active_robot_id() {
                let status = self.driver.update(&world, ctx);
                if let WaypointStatus::Failed(reason) = status {
                    self.planner.record_failure(active_id, reason, now);
                }
            }
        }

        // Compliance: name the active robot as the kicker during our restarts so the
        // executor exempts it and positions everyone else.
        if let Some(active_id) = self.driver.active_robot_id() {
            if let Some(role) = our_kicker_role(game_state, us_operating) {
                if let Some(p) = ctx.player(active_id) {
                    p.set_role(role);
                }
            }
        }

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

    /// Pick the nearest eligible robot to the ball as the set-piece kicker, send it
    /// to the ball, and name it so the executor exempts it from restart clamping.
    fn designate_prep_kicker(
        &self,
        world: &World,
        ctx: &mut TeamContext,
        game_state: GameState,
    ) -> Option<PlayerId> {
        let ball_pos = world.ball_position()?;
        let keeper_id = world.our_keeper_id();
        let kicker = world
            .own_players()
            .iter()
            .filter(|p| Some(p.id) != keeper_id)
            .min_by(|a, b| {
                let da = (a.position - ball_pos).norm();
                let db = (b.position - ball_pos).norm();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })?;
        let id = kicker.id;
        let role = if game_state == GameState::PrepareKickoff {
            "kickoff_kicker"
        } else {
            "penalty_kicker"
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
