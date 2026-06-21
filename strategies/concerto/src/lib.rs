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
pub mod possession;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use driver::{Driver, WaypointStatus};
use formation::Formation;
use planner::{PlanInputs, Planner};
use possession::{classify_raw, Possession, PossessionTracker};

/// The Concerto strategy.
pub struct ConcertoStrategy {
    tracker: PossessionTracker,
    planner: Planner,
    driver: Driver,
    formation: Formation,
    last_game_state: GameState,
    double_touch_robot: Option<PlayerId>,
}

impl ConcertoStrategy {
    pub fn new() -> Self {
        Self {
            tracker: PossessionTracker::new(),
            planner: Planner::new(),
            driver: Driver::new(),
            formation: Formation::new(),
            last_game_state: GameState::Unknown,
            double_touch_robot: None,
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
        // Owned snapshot so reads don't borrow `ctx` while we issue commands.
        let world = World::new(ctx.world().raw_snapshot().clone());
        let game_state = world.game_state();
        let us_operating = world.us_operating();
        let now = world.timestamp();

        // Runtime toggle: when on, skip the offensive loop so every field robot
        // stays Formation-controlled (for testing defence in isolation).
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

        // ── Possession (root stability surface) ─────────────────────────
        let raw = classify_raw(&world);
        let possession = self.tracker.update(raw, now);
        let possession_changed = self.tracker.changed_this_tick();

        let ball_present = world.ball_position().is_some();
        let we_may_act = match game_state {
            GameState::Run => true,
            GameState::Kickoff | GameState::FreeKick | GameState::PenaltyRun => us_operating,
            _ => false,
        };

        // ── Offensive loop: plan on events, then drive the active robot ──
        let mut plan_slots: Vec<PlayerId> = Vec::new();
        if !defense_only && ball_present && world.is_ball_in_play() && we_may_act {
            let needs_replan = self.planner.current_plan().is_none()
                || matches!(
                    self.driver.status(),
                    WaypointStatus::Succeeded | WaypointStatus::Failed(_)
                )
                || possession_changed
                || game_state_changed;

            if needs_replan {
                let inputs = PlanInputs {
                    keeper_id: world.our_keeper_id(),
                    double_touch_robot: self.double_touch_robot,
                    our_attacking_restart: us_operating
                        && matches!(game_state, GameState::Kickoff | GameState::FreeKick),
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
                plan_slots.push(active_id);
                let status = self.driver.update(&world, ctx);
                if let WaypointStatus::Failed(reason) = status {
                    self.planner.record_failure(active_id, reason, now);
                }
                // Kick feedforward: if we just fired a kick, tell possession the ball
                // is released so the next replan doesn't re-task the shooter.
                if let Some(kicker) = self.driver.take_kick_event() {
                    self.tracker.notify_release(kicker, now);
                }
                // Pass seam: a completed pass would hand off to the receiver here.
                if let Some(_next) = self.driver.take_new_active() {
                    // Unused in v1 (no waypoint sets it).
                }

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

        let plan_context = self.driver.plan_context_area();

        // ── Formation (all field robots except keeper + plan slots) ─────
        let commands = self
            .formation
            .update(&world, &plan_slots, plan_context, now);
        for cmd in &commands {
            if let Some(p) = ctx.player(cmd.id) {
                p.go_to(cmd.target).facing(cmd.face);
                p.set_role(cmd.role);
            }
        }

        // ── Goalkeeper ──────────────────────────────────────────────────
        if let Some(kid) = world.our_keeper_id() {
            let kpos = keeper::keeper_target(&world, config::KEEPER_DEPTH);
            let face = world
                .ball_position()
                .unwrap_or_else(|| world.opp_goal_center());
            if let Some(k) = ctx.player(kid) {
                k.go_to(kpos).facing(face);
                k.set_role("goalkeeper");
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
        };
        debug::string("possession", &poss_str);

        if let Some(active) = self.driver.active_robot_id() {
            if let Some(p) = world.own_player(active) {
                debug::circle("active_robot", p.position, 200.0);
            }
        }
        for cmd in commands {
            debug::cross_colored(
                &format!("formation_{}", cmd.id.as_u32()),
                cmd.target,
                DebugColor::Blue,
            );
        }
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
