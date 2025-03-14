// use num_traits::float::Float;
use std::collections::HashMap;

use dies_core::{Angle, BallData, PlayerData, PlayerId, RoleType, Vector2, WorldData};

use super::StrategyCtx;
use crate::{
    invoke_skill,
    roles::{
        skills::{ApproachBall, Face, FetchBallWithHeading, Kick},
        RoleCtx, SkillResult,
    },
    skill,
    strategy::{kickoff::OtherPlayer, Role, Strategy},
    PlayerControlInput,
};

pub struct FreeKickStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    kicker_id: Option<PlayerId>,
    keeper_id: Option<PlayerId>,
}

pub struct FreeAttacker {
    init_ball: Option<BallData>,
    target_direction: Option<Angle>,
}

impl Default for FreeAttacker {
    fn default() -> Self {
        Self::new()
    }
}

impl FreeAttacker {
    pub fn new() -> Self {
        Self {
            init_ball: None,
            target_direction: None,
        }
    }

    fn find_best_direction(
        &self,
        ball_pos: Vector2,
        player: &PlayerData,
        world: &WorldData,
    ) -> Angle {
        Angle::between_points(
            player.position,
            Vector2::new(4500.0, 100.0 * player.position.y.signum()),
        )
    }
}

impl Role for FreeAttacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let player_data = ctx.player;
        let world_data = ctx.world;

        let (_ball, init_ball) = if let Some(ball) = &world_data.ball {
            (ball, self.init_ball.get_or_insert(ball.clone()))
        } else {
            return PlayerControlInput::new();
        };
        let ball_pos = init_ball.position.xy();

        let target = if let Some(target) = self.target_direction {
            target
        } else {
            let target = self.find_best_direction(ball_pos, player_data, world_data);
            self.target_direction = Some(target);
            target
        };

        skill!(ctx, FetchBallWithHeading::new(target));

        loop {
            skill!(ctx, ApproachBall::new());
            let target = Vector2::new(
                4500.0,
                f64::max(f64::min(ctx.player.position.y, -400.0), 400.0),
            );
            match invoke_skill!(ctx, Face::towards_position(target).with_ball()) {
                crate::roles::SkillProgress::Continue(mut input) => {
                    input.with_dribbling(1.0);
                    return input;
                }
                _ => {}
            }
            if let SkillResult::Success = skill!(ctx, Kick::new()) {
                break;
            }
        }

        PlayerControlInput::new()
    }

    fn role_type(&self) -> RoleType {
        RoleType::FreeKicker
    }
}

impl FreeKickStrategy {
    pub fn new(keeper_id: Option<PlayerId>) -> Self {
        FreeKickStrategy {
            roles: HashMap::new(),
            kicker_id: None,
            keeper_id,
        }
    }

    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }

    pub fn set_gate_keeper(&mut self, id: PlayerId) {
        self.keeper_id = Some(id);
    }
}

impl Strategy for FreeKickStrategy {
    fn name(&self) -> &'static str {
        "FreeKick"
    }

    fn on_enter(&mut self, ctx: StrategyCtx) {
        // Clear roles
        self.roles.clear();
        self.kicker_id = None;

        log::info!(
            "Entering FreeKick strategy, game state: {:?}",
            ctx.world.current_game_state
        );

        // Assign player closest to the ball as kicker
        // if we are attacking
        if ctx.world.current_game_state.us_operating {
            let kicker = ctx
                .world
                .own_players
                .iter()
                .filter(|p| Some(p.id) != self.keeper_id)
                .min_by_key(|p| {
                    let ball_pos = ctx.world.ball.as_ref().unwrap().position.xy();
                    let diff = p.position - ball_pos;
                    diff.norm_squared() as i64
                });
            if let Some(kicker) = kicker {
                self.kicker_id = Some(kicker.id);
                log::info!("assigning {} as kicker", kicker.id);
                self.roles.insert(
                    kicker.id,
                    Box::new(FreeAttacker {
                        init_ball: None,
                        target_direction: None,
                    }),
                );
            } else {
                log::error!("No kicker found");
            }
        }

        // Assign roles to players
        for player_data in ctx.world.own_players.iter() {
            if self.keeper_id == Some(player_data.id) {
                continue;
            }
            if self.kicker_id == Some(player_data.id) {
                continue;
            }

            if let std::collections::hash_map::Entry::Vacant(e) = self.roles.entry(player_data.id) {
                let ball_pos = ctx.world.ball.as_ref().unwrap().position.xy();
                let distance = (player_data.position - ball_pos).norm();

                if distance < 650.0 {
                    let max_radius = 1000;
                    let min_distance = 700.0;
                    let target = dies_core::nearest_safe_pos(
                        dies_core::Avoid::Circle { center: ball_pos },
                        min_distance,
                        player_data.position,
                        player_data.position,
                        max_radius,
                        &ctx.world.field_geom.clone().unwrap_or_default(),
                    );

                    e.insert(Box::new(OtherPlayer::new(target)));
                }
            }
        }
    }

    fn update(&mut self, ctx: StrategyCtx) {}

    fn get_role(&mut self, player_id: PlayerId, ctx: StrategyCtx) -> Option<&mut dyn Role> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.as_mut())
        } else {
            None
        }
    }
}
