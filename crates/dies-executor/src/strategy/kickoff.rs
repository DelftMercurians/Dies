use std::collections::HashMap;
use log::log;
use dies_core::{Angle, GameState, PlayerData, PlayerId, WorldData};
use crate::strategy::{Role, Strategy};
use nalgebra::Vector2;
use crate::{PlayerControlInput, PlayerInputs};
use crate::roles::RoleCtx;
use crate::strategy::Task::{Task3Phase, Task4Phase};



struct PositionGenerator {
    d: f64,
    x: f64,
    y: f64,
    direction: f64,
    count: u32,
}

impl PositionGenerator {
    fn new(d: f64) -> Self {
        Self {
            d,
            x: d,
            y: 0.0,
            direction: 1.0,
            count: 1,
        }
    }
}

impl Iterator for PositionGenerator {
    type Item = Vector2<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        let result = Vector2::new(self.x, self.y);
        self.count += 1;
        if self.count % 2 == 0 {
            self.x += self.d;
            self.y = 0.0;
            self.direction = 1.0;
        } else {
            self.y = self.direction * (self.count as f64 / 2.0) * self.d;
            self.direction *= -1.0;
        }
        Some(result)
    }
}


pub struct KickoffStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    has_kicker: bool,
    gate_keeper_id: Option<PlayerId>,
    position_generator: PositionGenerator,
}


pub struct Kicker {
    move_to_ball: Task3Phase,
    move_to_circle: Task3Phase,
    kick: Task4Phase,
}

pub struct OtherPlayer {
    move_to_half_field: Task3Phase,
    fixed_position: Vector2<f64>,
}

impl Kicker {
    pub fn new() -> Self {
        Self {
            move_to_ball: Task3Phase::new(),
            kick: Task4Phase::new(),
            move_to_circle: Task3Phase::new(),
        }
    }
}

impl OtherPlayer {
    pub fn new(position: Vector2<f64>) -> Self {
        Self {
            move_to_half_field: Task3Phase::new(),
            fixed_position: position,
        }
    }
}

impl Role for Kicker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        
        let gamestate = ctx.world.current_game_state.game_state;
        let player_data = ctx.player;
        if gamestate == GameState::PrepareKickoff {
            return self.move_to_circle.relocate(player_data, Vector2::new(-100.0, 0.0), Angle::from_degrees(0.0));
        }
        
        if self.move_to_ball.is_accomplished() {
            //kick
            return self.kick.kick();
        }
        else if let Some(balldata) = ctx.world.ball.clone() {
            let ball_pos_v2 = Vector2::new(balldata.position.x, balldata.position.y);
            return self.move_to_ball.relocate(player_data, ball_pos_v2, Angle::from_degrees(0.0));
        } else {
            return PlayerControlInput::new();
        }
    }
}

impl Role for OtherPlayer {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        // log the player's position and the fixed position
        return self.move_to_half_field.relocate(ctx.player, self.fixed_position, Angle::from_degrees(0.0));
    }
}

impl KickoffStrategy {
    pub fn new(gate_keeper_id: Option<PlayerId>) -> Self {
        KickoffStrategy {
            roles: HashMap::new(),
            has_kicker: false,
            gate_keeper_id,
            position_generator: PositionGenerator::new(500.0),
        }
    }

    pub fn add_role_with_id(&mut self, id: PlayerId, role: Box<dyn Role>) {
        self.roles.insert(id, role);
    }
    
    pub fn set_gate_keeper(&mut self, id: PlayerId) {
        self.gate_keeper_id = Some(id);
    }
}

impl Strategy for KickoffStrategy {

    fn update(&mut self, world: &WorldData) -> PlayerInputs {
        let us_attacking = world.current_game_state.us_operating;
        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                if player_data.id == gate_keeper_id {
                    continue;
                }
            }
            if !self.roles.contains_key(&player_data.id) {
                if us_attacking && !self.has_kicker {
                    self.has_kicker = true;
                    self.roles.insert(player_data.id, Box::new(Kicker::new()));
                    log::info!("Adding player {} as the kicker", player_data.id);
                } else {
                    log::info!("Adding player {} as normal role", player_data.id);
                    self.roles.insert(player_data.id, Box::new(OtherPlayer::new(self.position_generator.next().unwrap())));
                }
            }
        }


        let mut inputs = PlayerInputs::new();
        for (id, role) in self.roles.iter_mut() {
            if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
                let player_data = player_data.clone();
                
                let input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
                inputs.insert(*id, input);
            } else {
                log::error!("No detetion data for player #{id} with active role");
            }
        }
        inputs
    }
}