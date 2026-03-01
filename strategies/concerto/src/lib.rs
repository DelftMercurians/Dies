pub mod driver;
pub mod formation;
pub mod geometry;
pub mod planner;
pub mod possession;

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

use driver::Driver;
use formation::Formation;
use planner::Planner;
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

        let ball_pos = match world.ball_position() {
            Some(p) => p,
            None => {
                // No ball detected — stop everyone
                for player in ctx.players() {
                    player.stop();
                    player.set_role("NoBall");
                }
                return;
            }
        };

        let possession = detect_possession(&world);
        let possession_category = PossessionCategory::from(&possession);

        // Determine if we need to replan
        let driver_status = self.driver.status().clone();
        let needs_replan = self.planner.current_plan().is_none()
            || matches!(
                driver_status,
                driver::WaypointStatus::Succeeded | driver::WaypointStatus::Failed(_)
            )
            || possession_category != self.last_possession_category;

        self.last_possession_category = possession_category;

        // Replan if needed
        if needs_replan {
            if let Some(plan) = self.planner.replan(&world, &possession) {
                self.driver.set_waypoint(
                    plan.waypoints[0].clone(),
                    plan.active_robot,
                    world.timestamp(),
                );
            }
        }

        // Collect IDs before mutable borrow of ctx
        let keeper_id = world.our_keeper_id();
        let active_robot_id = self.driver.active_robot_id();
        let plan_context = self.driver.plan_context_area();
        let own_player_ids: Vec<PlayerId> = ctx.player_ids().to_vec();

        // Run the driver — produces skill commands for the active robot
        let _waypoint_status = self.driver.update(&world, ctx);

        // Formation: compute roles and assign
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

        // Issue formation commands
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
