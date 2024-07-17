use crate::roles::Goalkeeper;
use crate::roles::RoleCtx;
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

        // find the ball
        if let Some(ball) = &world_data.ball {
            if self.init_ball.is_none() {
                self.init_ball = Some(ball.clone());
            }
        }
        if self.move_to_ball.is_accomplished() && gamestate == GameState::Penalty {
            //stage 3: kick
            if self.manipulating_ball.is_accomplished() {
                // find the ball
                if let Some(ball) = &world_data.ball {
                    if self.init_ball.is_none() {
                        self.init_ball = Some(ball.clone());
                    }
                }
                let ball_pos = self.init_ball.as_ref().unwrap().position.xy();
                println!("kicking in the ballls");
                return self.kick.kick(
                    PlayerControlInput::new()
                        .with_position(ball_pos)
                        .clone()
                );
            }
            //stage 2: dribbling
            else {
                let ball_pos = self.init_ball.as_ref().unwrap().position.xy();

                // find the goal keeper
                let goalkeeper_dir = world_data
                    .opp_players
                    .iter()
                    .find(|player| {
                        let pos = player.position;
                        pos.x >= 3500.0 && pos.x <= 4500.0 && pos.y >= -1000.0 && pos.y <= 1000.0
                    })
                    .map(|player| Angle::between_points(player_data.position, player.position));
                let target: Angle = goalkeeper_dir.map_or(Angle::from_radians(0.0), |dir| {
                    if dir.radians() > 0.0 {
                        (Angle::between_points(player_data.position, Vector2::new(4500.0, -1000.0))
                            + goalkeeper_dir.unwrap())
                            / 2.0
                    } else {
                        (Angle::between_points(player_data.position, Vector2::new(4500.0, 1000.0))
                            + goalkeeper_dir.unwrap())
                            / 2.0
                    }
                });
                return self.manipulating_ball.relocate(
                    player_data,
                    ball_pos.xy(),
                    target,
                    1.0,
                    0.0
                );
            }
        }
        // stage1: move to ball
        if let Some(_ball) = &world_data.ball {
            let ball_pos = self.init_ball.as_ref().unwrap().position.xy();
            let dir = Angle::between_points(player_data.position, ball_pos);
            let dist = (ball_pos - player_data.position).norm();
            let dirvec = (ball_pos - player_data.position) / dist;
            let goto_pos = player_data.position + dirvec * (dist - 100.0);
            self.move_to_ball
                .relocate(player_data, goto_pos, dir, 0.0, 0.5)
        } else {
            PlayerControlInput::new()
        }
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
    fn update(&mut self, ctx: StrategyCtx) {
        let world = ctx.world;
        let us_attacking = world.current_game_state.us_operating;

        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                if player_data.id == gate_keeper_id {
                    if !us_attacking {
                        self.roles
                            .insert(player_data.id, Box::new(Goalkeeper::new()));
                    }
                    continue;
                }
            }
            if let std::collections::hash_map::Entry::Vacant(e) = self.roles.entry(player_data.id) {
                if us_attacking && !self.has_attacker {
                    self.has_attacker = true;
                    e.insert(Box::new(Attacker::new()));
                } else if let Some(ball) = &world.ball {
                    let ball_pos = ball.position;
                    let pos_x = ball_pos.x + if us_attacking { -1000.0 } else { 1000.0 };
                    self.pos_counter += 1;
                    let mut pos_y = (self.pos_counter / 2) as f64 * self.pos_interval;
                    if self.pos_counter % 2 != 0 {
                        pos_y = -pos_y;
                    }
                    e.insert(Box::new(OtherPlayer::new(Vector2::new(pos_x, pos_y))));
                }
            }
        }

        // let mut inputs = PlayerInputs::new();
        // for (id, role) in self.roles.iter_mut() {
        //     if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
        //         let player_data = player_data.clone();

        //         let mut input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
        //         if world.current_game_state.game_state != PenaltyRun && self.gate_keeper_id.map_or(false, |id| id == player_data.id) {
        //             input = PlayerControlInput::new();
        //         }
        //         inputs.insert(*id, input);
        //     } else {
        //         log::error!("No detetion data for player #{id} with active role");
        //     }
        // }
        // inputs
    }

    fn update_role(
        &mut self,
        player_id: PlayerId,
        ctx: crate::roles::RoleCtx,
    ) -> Option<crate::PlayerControlInput> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.update(ctx))
        } else {
            None
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
