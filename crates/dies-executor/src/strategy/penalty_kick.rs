use crate::roles::RoleCtx;
use crate::strategy::Task::{Task3Phase, Task4Phase};
use crate::strategy::{Role, Strategy};
use crate::{PlayerControlInput, PlayerInputs};
use dies_core::{Angle, GameState, PlayerData, PlayerId, RoleType, WorldData};
use log::log;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::f64::consts::PI;
use dies_core::GameState::{PenaltyRun, PreparePenalty};
use crate::strategy::kickoff::{OtherPlayer};
use crate::roles::Goalkeeper;

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
}


impl Attacker {
    pub fn new() -> Self {
        Self {
            move_to_ball: Task3Phase::new(),
            manipulating_ball: Task3Phase::new(),
            kick: Task4Phase::new(),
        }
    }
}


impl Role for Attacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let gamestate = ctx.world.current_game_state.game_state;
        let player_data = ctx.player;
        let world_data = ctx.world;

        if self.move_to_ball.is_accomplished() && gamestate == GameState::Penalty {
            //stage 3: kick
            if self.manipulating_ball.is_accomplished(){
                self.kick.kick();
            }
                //stage 2: dribbling
            else {
                // find the goal keeper
                let goalkeeper_dir = world_data.opp_players.iter().find(|player|{
                    let pos = player.position;
                    pos.x >= 3500.0 && pos.x <= 4500.0 && pos.y >= -1000.0 && pos.y <= 1000.0
                }).map(|player|
                    (player.position - player_data.position).angle(&Vector2::new(1.0, 0.0))
                );
                let target = goalkeeper_dir.map_or(0.0, |dir| {
                    if dir > 0.0 {
                        (Vector2::new(4500.0, -1000.0) - player_data.position).angle(&Vector2::new(1.0, 0.0))
                    } else {
                        (Vector2::new(4500.0, 1000.0) - player_data.position).angle(&Vector2::new(1.0, 0.0))
                    }
                });
                self.manipulating_ball.relocate(player_data, player_data.position, Angle::from_radians(target), 1.0);
            }
        }
        // stage1: move to ball
        if let Some(ball) = &world_data.ball {
            let ball_pos = ball.position;
            let dir = (ball_pos.xy() - player_data.position).angle(&Vector2::new(1.0, 0.0));
            return self.move_to_ball.relocate(player_data, ball_pos.xy(), Angle::from_radians(dir), 0.0);
        } else {
            return PlayerControlInput::new();
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
    fn update(&mut self, world: &WorldData) -> PlayerInputs {
        let us_attacking = world.current_game_state.us_operating;

        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                if player_data.id == gate_keeper_id {
                    if (!us_attacking){
                        self.roles.insert(player_data.id, Box::new(Goalkeeper::new()));   
                    }
                    continue;
                }
            }
            if !self.roles.contains_key(&player_data.id) {
                if us_attacking && !self.has_attacker {
                    self.has_attacker = true;
                    self.roles.insert(player_data.id, Box::new(Attacker::new()));
                } else {
                    if let Some(ball) = &world.ball {
                        let ball_pos = ball.position;
                        let pos_x = ball_pos.x + if us_attacking { -1000.0 } else { 1000.0 };
                        self.pos_counter += 1;
                        let mut pos_y = (self.pos_counter / 2) as f64 * self.pos_interval;
                        if self.pos_counter % 2 != 0 {
                            pos_y = -pos_y;
                        }
                        self.roles.insert(
                            player_data.id,
                            Box::new(OtherPlayer::new(Vector2::new(pos_x,pos_y))),
                        );
                    } else {
                        return PlayerInputs::new();
                    }



                }
            }
        }

        let mut inputs = PlayerInputs::new();
        for (id, role) in self.roles.iter_mut() {
            if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
                let player_data = player_data.clone();

                let mut input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
                if (world.current_game_state.game_state != PenaltyRun && player_data.id == self.gate_keeper_id.unwrap()){
                    input = PlayerControlInput::new();
                }
                inputs.insert(*id, input);
            } else {
                log::error!("No detetion data for player #{id} with active role");
            }
        }
        inputs
    }

    fn get_role_type(&self, player_id: PlayerId) -> Option<RoleType> {
        self.roles.get(&player_id).map(|r| r.role_type())
    }
}
