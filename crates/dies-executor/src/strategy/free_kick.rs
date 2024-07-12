use crate::roles::RoleCtx;
use crate::strategy::task::{Task3Phase, Task4Phase};
use crate::strategy::{Role, Strategy};
use crate::{PlayerControlInput, PlayerInputs};
use dies_core::{Angle, BallData, GameState, PlayerData, PlayerId, RoleType, WorldData};
use log::log;
use nalgebra::Vector2;
use std::collections::HashMap;
use std::f64::consts::PI;
use crate::strategy::kickoff::{OtherPlayer};
use crate::roles::Goalkeeper;

pub struct FreeKickStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    has_attacker: bool,
    gate_keeper_id: Option<PlayerId>

}

pub struct FreeAttacker {
    move_to_ball: Task3Phase,
    manipulating_ball: Task3Phase,
    kick: Task4Phase,
    init_ball: Option<BallData>,
}


impl FreeAttacker {
    pub fn new() -> Self {
        Self {
            move_to_ball: Task3Phase::new(),
            manipulating_ball: Task3Phase::new(),
            kick: Task4Phase::new(),
            init_ball: None,
        }
    }
}


impl Role for FreeAttacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let gamestate = ctx.world.current_game_state.game_state;
        let player_data = ctx.player;
        let world_data = ctx.world;

        if self.move_to_ball.is_accomplished() {
            //stage 3: kick
            if self.manipulating_ball.is_accomplished(){
                println!("stage 3: kick");
                return self.kick.kick();
            }
                //stage 2: dribbling
            else {
                // find all possible dirs, it can directly shoot to the goal or pass to other player
                let mut dirs: Vec<Angle> = vec![];
                dirs.push(Angle::between_points(player_data.position, Vector2::new(4500.0, 0.0)));
                for own_player in world_data.own_players.iter() {
                    if own_player.id == player_data.id  {
                        continue;
                    }
                    dirs.push(Angle::between_points(player_data.position, own_player.position));
                }
                // find one that is closest to the current orientation

                let mut target = Angle::from_radians(0.0);
                let mut min_diff = player_data.yaw.radians().abs();
                for dir in dirs {
                    let diff = (dir - player_data.yaw).radians().abs();
                    if diff < min_diff {
                        min_diff = diff;
                        target = dir;
                    }
                }
                return self.manipulating_ball.relocate(player_data, player_data.position, target, 1.0);
            }
        }
        // stage1: move to ball
        if let Some(ball) = &world_data.ball {

            println!("stage1: {}", self.move_to_ball.is_accomplished());
            if self.init_ball.is_none() {
                self.init_ball = Some(ball.clone());
            }
            let ball_pos = self.init_ball.as_ref().unwrap().position;
            let dir= Angle::between_points(ball_pos.xy(), player_data.position);
            return self.move_to_ball.relocate(player_data, ball_pos.xy(), dir, 0.0);
        } else {
            return PlayerControlInput::new();
        }
    }
    fn role_type(&self) -> RoleType {
        RoleType::FreeKicker
    }
}


impl FreeKickStrategy {
    pub fn new(gate_keeper_id: Option<PlayerId>) -> Self {
        FreeKickStrategy {
            roles: HashMap::new(),
            has_attacker: false,
            gate_keeper_id,
        }
    }

    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }

    pub fn set_gate_keeper(&mut self, id: PlayerId) {
        self.gate_keeper_id = Some(id);
    }
}

impl Strategy for FreeKickStrategy {
    fn update(&mut self, world: &WorldData) -> PlayerInputs {
        let us_attacking = world.current_game_state.us_operating;

        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                    continue;
            }
            if !self.roles.contains_key(&player_data.id) {
                if us_attacking  {
                    if (!self.has_attacker){
                        self.has_attacker = true;
                        self.roles.insert(player_data.id, Box::new(FreeAttacker::new()));
                    }
                } else {
                    if let Some(ball) = &world.ball {
                        let ball_pos = ball.position;
                        // get the disance between the ball and the player
                        let distance = (player_data.position - ball_pos.xy()).norm();
                        if distance > 500.0 {
                            // get the target pos that is 500.0 away from the ball
                            let target = ball_pos.xy() + (player_data.position - ball_pos.xy()).normalize() * 500.0;
                            self.roles.insert(player_data.id, Box::new(OtherPlayer::new(target)));
                        }
                    }
                }
            }
        }

        let mut inputs = PlayerInputs::new();
        println!("game state: {:?}", world.current_game_state.game_state);
        for (id, role) in self.roles.iter_mut() {
            if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
                let player_data = player_data.clone();
                let mut input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
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
