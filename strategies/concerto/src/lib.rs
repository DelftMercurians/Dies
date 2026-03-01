pub mod driver;
pub mod formation;
pub mod geometry;
pub mod planner;
pub mod possession;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use driver::Driver;
use formation::Formation;
use planner::{PlanContext, Planner};
use possession::{detect_possession, PossessionCategory};

/// Main Concerto strategy struct implementing the Strategy trait.
pub struct ConcertoStrategy {
    planner: Planner,
    driver: Driver,
    formation: Formation,
    field_half_length: f64,
    field_half_width: f64,
    goal_width: f64,
    last_possession_category: PossessionCategory,
    last_game_state: GameState,
    double_touch_robot: Option<PlayerId>,
}

impl ConcertoStrategy {
    pub fn new() -> Self {
        Self {
            planner: Planner::new(),
            driver: Driver::new(),
            formation: Formation::new(),
            field_half_length: 4500.0,
            field_half_width: 3000.0,
            goal_width: 1000.0,
            last_possession_category: PossessionCategory::Loose,
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
    fn init(&mut self, world: &World) {
        self.field_half_length = world.field_length() / 2.0;
        self.field_half_width = world.field_width() / 2.0;
        self.goal_width = world.goal_width();
        tracing::info!("Concerto strategy initialized");
    }

    fn update(&mut self, ctx: &mut TeamContext) {
        // Clone the world snapshot so we have an owned World that doesn't borrow ctx.
        let world = World::new(ctx.world().raw_snapshot().clone());

        // Update cached field geometry
        self.field_half_length = world.field_length() / 2.0;
        self.field_half_width = world.field_width() / 2.0;
        self.goal_width = world.goal_width();

        let game_state = world.game_state();
        let us_operating = world.us_operating();

        // ── 1. Update double-touch tracking ─────────────────────────────
        if let Some(kicker) = world.freekick_kicker() {
            self.double_touch_robot = Some(kicker);
        } else if matches!(game_state, GameState::Halt | GameState::Stop | GameState::Timeout) {
            self.double_touch_robot = None;
        } else if game_state == GameState::Run && world.freekick_kicker().is_none() {
            // Framework cleared the kicker (another robot touched the ball)
            self.double_touch_robot = None;
        }

        // ── 2. Game state transition → clear plan + driver ──────────────
        let game_state_changed = game_state != self.last_game_state;
        if game_state_changed {
            self.planner.clear_plan();
            self.driver.clear();
        }
        self.last_game_state = game_state;

        // ── 3. Get ball position ────────────────────────────────────────
        let ball_pos = match world.ball_position() {
            Some(p) => p,
            None => {
                for player in ctx.players() {
                    player.stop();
                    player.set_role("NoBall");
                }
                return;
            }
        };

        // ── 4. Build PlanContext ─────────────────────────────────────────
        let our_set_piece = us_operating
            && matches!(
                game_state,
                GameState::Kickoff | GameState::FreeKick
            );
        let plan_ctx = PlanContext {
            our_set_piece,
            double_touch_robot: self.double_touch_robot,
        };

        // ── 5. Halt/Unknown → bail early (framework handles stopping) ───
        if matches!(game_state, GameState::Halt | GameState::Unknown | GameState::Timeout) {
            return;
        }

        // ── 6. Detect possession ────────────────────────────────────────
        let possession = detect_possession(&world);
        let possession_category = PossessionCategory::from(&possession);

        // ── 7. Replan triggers ──────────────────────────────────────────
        let driver_status = self.driver.status().clone();
        let needs_replan = self.planner.current_plan().is_none()
            || matches!(
                driver_status,
                driver::WaypointStatus::Succeeded | driver::WaypointStatus::Failed(_)
            )
            || possession_category != self.last_possession_category
            || game_state_changed;

        self.last_possession_category = possession_category;

        // ── 8. Replan if needed ─────────────────────────────────────────
        if needs_replan {
            if let Some(plan) = self.planner.replan(&world, &possession, &plan_ctx) {
                self.driver.set_waypoint(
                    plan.waypoints[0].clone(),
                    plan.active_robot,
                    world.timestamp(),
                );
            }
        }

        // ── 9. Collect IDs before mutable borrow of ctx ─────────────────
        let keeper_id = world.our_keeper_id();
        let active_robot_id = self.driver.active_robot_id();
        let plan_context = self.driver.plan_context_area();
        let own_player_ids: Vec<PlayerId> = ctx.player_ids().to_vec();

        // Run the driver — produces skill commands for the active robot
        let _waypoint_status = self.driver.update(&world, ctx);

        // ── 10. Override active robot role name for compliance ───────────
        if let Some(active_id) = active_robot_id {
            let role_override = match game_state {
                GameState::Kickoff | GameState::PrepareKickoff if us_operating => {
                    Some("kickoff_kicker")
                }
                GameState::FreeKick if us_operating => Some("free_kick_kicker"),
                _ => None,
            };
            if let Some(role_name) = role_override {
                if let Some(player) = ctx.player(active_id) {
                    player.set_role(role_name);
                }
            }
        }

        // ── 11. Handle PrepareKickoff (ours): designate a kicker ────────
        // During PrepareKickoff, there's no ball to capture yet, but we need
        // a robot with the kickoff_kicker role so it's exempt from own-half clamping.
        if game_state == GameState::PrepareKickoff && us_operating && active_robot_id.is_none() {
            // Pick the closest robot to the center (ball position) as the designated kicker
            let kicker = world
                .own_players()
                .iter()
                .filter(|p| Some(p.id) != keeper_id)
                .min_by(|a, b| {
                    let da = (a.position - ball_pos).norm();
                    let db = (b.position - ball_pos).norm();
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                });
            if let Some(k) = kicker {
                let kicker_id = k.id;
                if let Some(player) = ctx.player(kicker_id) {
                    player.go_to(ball_pos).facing(ball_pos);
                    player.set_role("kickoff_kicker");
                }
            }
        }

        // ── 12. Formation + goalkeeper (unchanged) ──────────────────────
        let formation_robot_ids: Vec<PlayerId> = own_player_ids
            .iter()
            .copied()
            .filter(|id| Some(*id) != keeper_id && Some(*id) != active_robot_id)
            .collect();

        let roles = self.formation.compute_roles(
            &world,
            plan_context,
            self.field_half_length,
            self.field_half_width,
        );

        let assignments =
            self.formation
                .assign_roles(&formation_robot_ids, &roles, &world);

        for (player_id, role) in &assignments {
            if let Some(player) = ctx.player(*player_id) {
                player.go_to(role.position).facing(ball_pos);
                player.set_role(role.name);
            }
        }

        // Goalkeeper
        if let Some(kid) = keeper_id {
            if let Some(keeper) = ctx.player(kid) {
                let keeper_pos = self.formation.compute_goalkeeper_position(
                    ball_pos,
                    self.field_half_length,
                    self.goal_width,
                );
                keeper.go_to(keeper_pos).facing(ball_pos);
                keeper.set_role("Goalkeeper");
            }
        }

        // Debug visualization
        dies_strategy_api::debug::cross("ball", ball_pos);
        if let Some(ctx_area) = plan_context {
            dies_strategy_api::debug::cross_colored(
                "plan_context",
                ctx_area,
                DebugColor::Yellow,
            );
        }
        for (i, role) in roles.iter().enumerate() {
            let key = format!("role_{}", i);
            dies_strategy_api::debug::cross_colored(&key, role.position, DebugColor::Blue);
        }
        if let Some(active_id) = active_robot_id {
            if let Some(p) = world.own_player(active_id) {
                dies_strategy_api::debug::circle("active_robot", p.position, 200.0);
            }
        }
    }
}
