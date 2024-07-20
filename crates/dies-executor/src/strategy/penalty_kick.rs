use crate::invoke_skill;
use crate::roles::skills::ApproachBall;
use crate::roles::skills::Face;
use crate::roles::skills::FetchBallWithHeading;
use crate::roles::skills::GoToPosition;
use crate::roles::skills::Kick;
use crate::roles::Goalkeeper;
use crate::roles::RoleCtx;
use crate::roles::SkillResult;
use crate::skill;
use crate::strategy::kickoff::OtherPlayer;
use crate::strategy::task::{Task3Phase, Task4Phase};
use crate::strategy::{Role, Strategy};
use crate::PlayerControlInput;
use dies_core::{Angle, BallData, GameState, PlayerId, RoleType};
use nalgebra::Vector2;
use std::collections::HashMap;

use super::StrategyCtx;

pub struct PenaltyKickStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    has_attacker: bool,
    pos_interval: f64,
    pos_counter: u32,
}

pub struct Attacker {
    move_to_ball: Task3Phase,
    manipulating_ball: Task3Phase,
    kick: Task4Phase,
    init_ball: Option<BallData>,
}

impl Default for Attacker {
    fn default() -> Self {
        Self::new()
    }
}

impl Attacker {
    pub fn new() -> Self {
        Self {
            move_to_ball: Task3Phase::new(),
            manipulating_ball: Task3Phase::new(),
            kick: Task4Phase::new(),
            init_ball: None,
        }
    }
}

impl Role for Attacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let gamestate = ctx.world.current_game_state.game_state;
        let player_data = ctx.player;
        let world_data = ctx.world;

        if let Some(ball) = &world_data.ball {
            if self.init_ball.is_none() {
                self.init_ball = Some(ball.clone());
            }
        }

        if gamestate == GameState::PreparePenalty {
            skill!(
                ctx,
                FetchBallWithHeading::towards_position(Vector2::new(4500.0, 0.0))
            );

            return PlayerControlInput::new();
        } else if gamestate == GameState::PenaltyRun || gamestate == GameState::Penalty {
            let ball_pos = self.init_ball.as_ref().unwrap().position.xy();

            // skill!(
            //     ctx,
            //     GoToPosition::new(Vector2::new(ball_pos.x + 1000.0, ball_pos.y))
            // );

            // find the goal keeper
            let goalkeeper_dir = world_data
                .opp_players
                .iter()
                .find(|player| {
                    let pos = player.position;
                    pos.x >= 3000.0 && pos.y >= -1000.0 && pos.y <= 1000.0
                })
                .map(|player| Angle::between_points(player_data.position, player.position));
            let target: Angle = goalkeeper_dir.map_or(Angle::from_radians(0.0), |dir| {
                if dir.radians() > 0.0 {
                    (Angle::between_points(player_data.position, Vector2::new(4500.0, 350.0))
                        + goalkeeper_dir.unwrap())
                        / 2.0
                } else {
                    (Angle::between_points(player_data.position, Vector2::new(4500.0, -350.0))
                        + goalkeeper_dir.unwrap())
                        / 2.0
                }
            });

            loop {
                skill!(ctx, ApproachBall::new());
                match invoke_skill!(ctx, Face::new(target).with_ball()) {
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
        }

        PlayerControlInput::new()
    }
    fn role_type(&self) -> RoleType {
        RoleType::PenaltyKicker
    }
}

impl PenaltyKickStrategy {
    pub fn new() -> Self {
        PenaltyKickStrategy {
            roles: HashMap::new(),
            has_attacker: false,
            pos_counter: 0,
            pos_interval: 500.0,
        }
    }
}

impl Strategy for PenaltyKickStrategy {
    fn name(&self) -> &'static str {
        "PenaltyKick"
    }

    fn on_enter(&mut self, ctx: StrategyCtx) {
        self.roles.clear();
        self.has_attacker = false;
        self.pos_counter = 0;
        let us_attacking = ctx.world.current_game_state.us_operating;

        let mut player_ids = ctx
            .world
            .own_players
            .iter()
            .map(|p| p.id)
            .collect::<Vec<PlayerId>>();
        player_ids.sort(); // increasing order, we take the last player as the attacker

        if us_attacking {
            if let Some(id) = player_ids.pop() {
                self.roles.insert(id, Box::new(Attacker::new()));
            }
        }

        if let Some(id) = player_ids.get(0) {
            self.roles.insert(*id, Box::new(Goalkeeper::new()));
            player_ids.remove(0);
        }

        for id in player_ids {
            // line up the players at x =
            let geom = ctx.world.field_geom.clone().unwrap_or_default();
            let fl = geom.field_length / 2.0;
            let pw = geom.penalty_area_width / 2.0;
            let pos = Vector2::new(
                if us_attacking { -fl } else { fl },
                pw - id.as_u32() as f64 * 300.0,
            );
            self.roles.insert(id, Box::new(OtherPlayer::new(pos)));
        }
    }

    fn update(&mut self, ctx: StrategyCtx) {}

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.as_mut())
        } else {
            None
        }
    }
}
