//! Concerto — a formation/planner-split RoboCup SSL strategy.
//!
//! Two layers above the skills:
//! - **Formation** positions every field robot except the keeper and the active
//!   (plan-controlled) robot.
//! - **Planner → Driver** moves the ball toward the opponent goal via
//!   ball-state-transition waypoints, re-deciding only on discrete events.

pub mod config;
pub mod driver;
pub mod formation;
pub mod geometry;
pub mod keeper;
pub mod matching;
pub mod planner;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use driver::{Driver, WaypointStatus};
use formation::Formation;
use planner::{PlanInputs, Planner};

/// The Concerto strategy.
pub struct ConcertoStrategy {
    /// Previous frame's possession, for detecting changes (the metric itself is
    /// the framework's single source of truth — `World::possession`).
    last_possession: Possession,
    planner: Planner,
    driver: Driver,
    formation: Formation,
    last_game_state: GameState,
    double_touch_robot: Option<PlayerId>,
    /// Ball position when we gained possession (the dribble contact point), for the
    /// excessive-dribbling carry cap. Cleared whenever we don't hold the ball.
    dribble_origin: Option<Vector2>,
    /// Goalkeeper Guard/Clear state machine.
    keeper: keeper::KeeperState,
}

impl ConcertoStrategy {
    pub fn new() -> Self {
        Self {
            last_possession: Possession::Loose,
            planner: Planner::new(),
            driver: Driver::new(),
            formation: Formation::new(),
            last_game_state: GameState::Unknown,
            double_touch_robot: None,
            dribble_origin: None,
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
        vec![ParamSpec::bool(
            "defense_only",
            "Defense only (suppress offense)",
            false,
        )]
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

        // ── Hard stops: let the executor halt the robots ────────────────
        if matches!(
            game_state,
            GameState::Halt | GameState::Unknown | GameState::Timeout
        ) {
            return;
        }

        let possession = world.possession();
        let possession_changed = possession != self.last_possession;
        self.last_possession = possession;

        // Track the dribble contact point: stamp it when we gain the ball, clear it
        // when we don't hold it. `carried` = how far we've dribbled since contact.
        match possession {
            Possession::We(_) => {
                if self.dribble_origin.is_none() {
                    self.dribble_origin = world.ball_position();
                }
            }
            _ => self.dribble_origin = None,
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

        // ── Offensive loop: plan on events, then drive the active robot ──
        let mut plan_slots: Vec<PlayerId> = Vec::new();
        if !defense_only && ball_present && world.is_ball_in_play() && we_may_act {
            // A pass is atomic: possession flits We→Loose→We while the ball is in
            // flight, so soft (possession/game-state) replan triggers are suppressed
            // mid-pass — the coordinator owns abort/failure and surfaces it as a
            // terminal status, which still replans via the arm below.
            let passing = self.driver.is_passing();
            let needs_replan = self.planner.current_plan().is_none()
                || matches!(
                    self.driver.status(),
                    WaypointStatus::Succeeded | WaypointStatus::Failed(_)
                )
                || (!passing && (possession_changed || game_state_changed));

            if needs_replan {
                let inputs = PlanInputs {
                    keeper_id: world.our_keeper_id(),
                    double_touch_robot: self.double_touch_robot,
                    our_attacking_restart: us_operating
                        && matches!(game_state, GameState::Kickoff | GameState::FreeKick),
                    carried,
                    now,
                };
                match self.planner.replan(&world, &possession, &inputs) {
                    Some(plan) => {
                        self.driver
                            .set_waypoint(plan.waypoints[0].clone(), plan.active_robot, now)
                    }
                    None => self.driver.clear(),
                }
            }

            if let Some(active_id) = self.driver.active_robot_id() {
                // Reserve every plan-controlled robot (both passer and receiver for
                // a pass) so Formation never commands them — a stray go_to would
                // cancel the pass.
                plan_slots.extend(self.driver.plan_slots());
                let status = self.driver.update(&world, ctx);
                if let WaypointStatus::Failed(reason) = status {
                    self.planner.record_failure(active_id, reason, now);
                }
                // A completed pass hands the ball to the receiver. We rely on the
                // Succeeded status above to trigger a replan that picks up the new
                // carrier via possession; the hint is consumed to keep it fresh.
                let _ = self.driver.take_new_active();

                // Compliance: name the active robot as the kicker during our restarts
                // so the executor exempts it and positions everyone else.
                if let Some(role) = our_kicker_role(game_state, us_operating) {
                    if let Some(p) = ctx.player(active_id) {
                        p.set_role(role);
                    }
                }
            }
        } else {
            // Not acting offensively — replan fresh when play resumes.
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
                plan_slots.push(id);
            }
        }

        let plan_context = self.driver.plan_context();

        // ── Formation (all field robots except keeper + plan slots) ─────
        let commands = self
            .formation
            .update(&world, &plan_slots, &plan_context, now);
        for cmd in &commands {
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

        self.draw_debug(&world, &possession, &commands);
    }
}

impl ConcertoStrategy {
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
        if let Some(p) = ctx.player(id) {
            p.go_to(ball_pos).facing(world.opp_goal_center());
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
    use planner::{CaptureKind, Waypoint};
    let (kind, detail) = match wp {
        Waypoint::Capture { kind, robot } => {
            let what = match kind {
                CaptureKind::Loose => "loose ball".to_string(),
                CaptureKind::Steal { from } => format!("steal from p{}", from.as_u32()),
            };
            ("Capture", format!("p{}: {what}", robot.as_u32()))
        }
        Waypoint::Dribble { target_area } => (
            "Dribble",
            format!("→ ({:.0}, {:.0})", target_area.x, target_area.y),
        ),
        Waypoint::Shoot { target } => ("Shoot", format!("→ ({:.0}, {:.0})", target.x, target.y)),
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
