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
    gate_keeper_id: Option<PlayerId>,
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
    pub fn new(gate_keeper_id: Option<PlayerId>) -> Self {
        PenaltyKickStrategy {
            roles: HashMap::new(),
            has_attacker: false,
            gate_keeper_id,
            pos_counter: 0,
            pos_interval: 500.0,
        }
    }

    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }

    pub fn set_gate_keeper(&mut self, id: PlayerId) {
        self.gate_keeper_id = Some(id);
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

        // Assign attacker role
        for player in ctx.world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                println!("assigning {} as gate keeper", gate_keeper_id);
                self.roles
                    .insert(gate_keeper_id, Box::new(Goalkeeper::new()));
            }
            if !self.has_attacker && ctx.world.current_game_state.us_operating {
                println!("assigning {} as attacker", player.id);
                self.has_attacker = true;
                self.roles.insert(player.id, Box::new(Attacker::new()));
            } else {
                println!("assigning {} as other player", player.id);
                self.roles.insert(
                    player.id,
                    Box::new(OtherPlayer::new(Vector2::new(
                        200.0 + (player.id.as_u32() as f64 * ctx.world.player_model.radius + 50.0),
                        -ctx.world
                            .field_geom
                            .as_ref()
                            .map(|g| g.field_width / 2.0)
                            .unwrap_or(-3000.0),
                    ))),
                );
            }
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
