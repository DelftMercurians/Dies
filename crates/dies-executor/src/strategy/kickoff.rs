use crate::roles::skills::{FetchBall, GoToPosition, Kick};
use crate::roles::RoleCtx;
use crate::strategy::{Role, Strategy};
use crate::{skill, PlayerControlInput};
use dies_core::{Angle, GameState, PlayerId, RoleType};
use nalgebra::Vector2;
use std::collections::HashMap;
use std::f64::consts::PI;

use super::StrategyCtx;

/// A generator for creating a sequence of 2D positions.
///
/// This struct generates positions in a specific pattern, alternating between
/// moving right and moving up/down, with increasing vertical distances.
#[derive(Debug)]
struct PositionGenerator {
    /// The base distance used for position calculations.
    d: f64,
    /// The current x-coordinate.
    x: f64,
    /// The current y-coordinate.
    y: f64,
    /// The current vertical direction (1.0 for up, -1.0 for down).
    direction: f64,
    /// The current count of generated positions.
    count: u32,
}

impl PositionGenerator {
    /// Create a new position generator with the given base distance.
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
        self.count += 1;
        if self.count % 2 == 0 {
            self.x += self.d;
            self.direction = 1.0;
        } else {
            self.direction *= -1.0;
        }
        self.y = self.direction * ((self.count as f64 / 2.0) + 1.0) * self.d;
        Some(Vector2::new(self.x, self.y))
    }
}

pub struct KickoffStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    has_kicker: bool,
    gate_keeper_id: Option<PlayerId>,
    position_generator: PositionGenerator,
}

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
                    skill!(ctx, FetchBall::new());
                    skill!(ctx, Kick::new());
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
            GoToPosition::new(self.fixed_position).with_heading(Angle::from_degrees(90.0)).avoid_ball()
        );

        PlayerControlInput::new()
    }

    fn role_type(&self) -> RoleType {
        RoleType::Player
    }
}

impl KickoffStrategy {
    pub fn new(gate_keeper_id: Option<PlayerId>) -> Self {
        KickoffStrategy {
            roles: HashMap::new(),
            has_kicker: false,
            gate_keeper_id,
            position_generator: PositionGenerator::new(-1000.0),
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
    fn name(&self) -> &'static str {
        "Kickoff"
    }

    fn on_enter(&mut self, _ctx: StrategyCtx) {
        // Clear roles
        self.has_kicker = false;
        self.roles.clear();
    }

    fn update(&mut self, ctx: StrategyCtx) {
        let world = ctx.world;

        let us_attacking = world.current_game_state.us_operating;
        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                if player_data.id == gate_keeper_id {
                    continue;
                }
            }
            if let std::collections::hash_map::Entry::Vacant(e) = self.roles.entry(player_data.id) {
                if !self.has_kicker {
                    self.has_kicker = true;
                    e.insert(Box::new(Kicker::new(us_attacking)));
                    log::info!("Adding player {} as the kicker", player_data.id);
                } else {
                    log::info!("Adding player {} as normal role", player_data.id);
                    e.insert(Box::new(OtherPlayer::new(
                        self.position_generator.next().unwrap(),
                    )));
                }
            }
        }

        // let mut inputs = PlayerInputs::new();
        // for (id, role) in self.roles.iter_mut() {
        //     if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
        //         let player_data = player_data.clone();

        //         let input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
        //         inputs.insert(*id, input);
        //     } else {
        //         log::error!("No detetion data for player #{id} with active role");
        //     }
        // }
        // inputs
    }

    fn get_role(&mut self, player_id: PlayerId) -> Option<&mut dyn Role> {
        if let Some(role) = self.roles.get_mut(&player_id) {
            Some(role.as_mut())
        } else {
            None
        }
    }
}
