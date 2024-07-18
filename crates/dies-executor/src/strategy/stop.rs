use std::collections::HashMap;

use dies_core::PlayerId;

use crate::roles::Role;

use super::Strategy;

struct StopRole {}

impl Role for StopRole {
    fn role_type(&self) -> dies_core::RoleType {
        dies_core::RoleType::Player
    }

    fn update(&mut self, ctx: crate::roles::RoleCtx<'_>) -> crate::PlayerControlInput {
        let mut input = crate::PlayerControlInput::new();
        input.with_speed_limit(1300.0);
        input.avoid_ball = true;
        if let Some(ball) = ctx.world.ball.as_ref() {
            let ball_pos = ball.position.xy();
            let dist = (ball_pos - ctx.player.position.xy()).norm();
            if dist < 560.0 {
                // Move away from the ball
                let target = ball_pos.xy()
                            + (ctx.player.position - ball_pos.xy()).normalize() * 650.0;
                input.with_position(target);
            }
        }
        input
    }
}

pub struct StopStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
}

impl StopStrategy {
    pub fn new() -> Self {
        Self {
            roles: HashMap::new(),
        }
    }
}

impl Strategy for StopStrategy {
    fn name(&self) -> &'static str {
        "Stop"
    }

    fn update(&mut self, ctx: super::StrategyCtx) {
        for p in ctx.world.own_players.iter() {
            if !self.roles.contains_key(&p.id) {
                self.roles.insert(p.id, Box::new(StopRole {}));
            }
        }
    }

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.as_mut())
        } else {
            None
        }
    }
}