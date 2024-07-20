use crate::roles::skills::{ApproachBall, FetchBall, GoToPosition, Kick};
use crate::roles::waller::Waller;
use crate::roles::{Goalkeeper, RoleCtx, SkillResult};
use crate::strategy::{Role, Strategy};
use crate::{skill, PlayerControlInput};
use dies_core::{Angle, GameState, PlayerId, RoleType};
use nalgebra::Vector2;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::StrategyCtx;

pub struct Kicker {
    us_attacking: bool,
}

pub struct OtherPlayer {
    fixed_position: Vector2<f64>,
}

impl Kicker {
    pub fn new(us_attacking: bool) -> Self {
        Self { us_attacking }
    }
}

impl OtherPlayer {
    pub fn new(position: Vector2<f64>) -> Self {
        Self {
            fixed_position: position,
        }
    }
}

impl Role for Kicker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        match ctx.world.current_game_state.game_state {
            GameState::PrepareKickoff => {
                skill!(
                    ctx,
                    GoToPosition::new(Vector2::new(-800.0, PI))
                        .with_heading(Angle::from_degrees(0.0))
                );
            }
            GameState::Kickoff => {
                if self.us_attacking {
                    loop {
                        skill!(ctx, ApproachBall::new());
                        if let SkillResult::Success = skill!(ctx, Kick::new()) {
                            break;
                        }
                    }
                }
            }
            _ => {}
        }

        PlayerControlInput::new()
    }
    fn role_type(&self) -> RoleType {
        RoleType::KickoffKicker
    }
}

impl Role for OtherPlayer {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        skill!(
            ctx,
            GoToPosition::new(self.fixed_position)
                .with_heading(Angle::from_degrees(90.0))
                .avoid_ball()
        );

        PlayerControlInput::new()
    }

    fn role_type(&self) -> RoleType {
        RoleType::Player
    }
}

pub struct KickoffStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
}

impl KickoffStrategy {
    pub fn new() -> Self {
        KickoffStrategy {
            roles: HashMap::new(),
        }
    }
}

impl Strategy for KickoffStrategy {
    fn name(&self) -> &'static str {
        "Kickoff"
    }

    fn on_enter(&mut self, ctx: StrategyCtx) {
        // Clear roles
        self.roles.clear();

        let mut player_ids = ctx
            .world
            .own_players
            .iter()
            .map(|p| p.id)
            .collect::<Vec<_>>();
        player_ids.sort();
        player_ids.reverse();

        if player_ids.len() == 1 {
            // Just one player, make him the kicker
            self.roles.insert(
                player_ids.pop().unwrap(),
                Box::new(Kicker::new(ctx.world.current_game_state.us_operating)),
            );
            return;
        }

        // 0 -> goalkeeper
        if let Some(id) = player_ids.pop() {
            self.roles.insert(id, Box::new(Goalkeeper::new()));
        }

        // 5 -> Kicker
        if let Some(id) = player_ids.get(0) {
            self.roles.insert(
                id.clone(),
                Box::new(Kicker::new(ctx.world.current_game_state.us_operating)),
            );
            player_ids.remove(0);
        }

        // 1 -> Waller
        if let Some(id) = player_ids.pop() {
            self.roles.insert(id, Box::new(Waller::new_with_index(0)));
        }

        // 2 -> Waller
        if let Some(id) = player_ids.pop() {
            self.roles.insert(id, Box::new(Waller::new_with_index(1)));
        }

        // 3+ -> Other players
        let mut other_positions = vec![
            Vector2::new(-1500.0, 400.0),
            Vector2::new(-1500.0, 400.0),
            Vector2::new(-2000.0, 0.0),
        ];
        for id in player_ids {
            self.roles.insert(
                id,
                Box::new(OtherPlayer::new(
                    other_positions.pop().unwrap_or(Vector2::new(-3500.0, 0.0)),
                )),
            );
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
