use crate::roles::skills::{ApproachBall, FetchBallWithHeading, Kick};
use crate::roles::RoleCtx;
use crate::strategy::kickoff::OtherPlayer;
use crate::strategy::task::{Task3Phase, Task4Phase};
use crate::strategy::{Role, Strategy};
use crate::{skill, PlayerControlInput};
use dies_core::{Angle, BallData, PlayerData, PlayerId, RoleType, Vector2, WorldData};
use std::collections::HashMap;

use super::StrategyCtx;

pub struct FreeKickStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    has_attacker: bool,
    gate_keeper_id: Option<PlayerId>,
}

pub struct FreeAttacker {
    move_to_ball: Task3Phase,
    kick: Task4Phase,
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
            move_to_ball: Task3Phase::new(),
            kick: Task4Phase::new(),
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
        let mut dirs: Vec<Angle> = vec![];
        let goaldir = Angle::between_points(ball_pos, Vector2::new(4500.0, 0.0));
        let our_goaldir = Angle::between_points(player.position, Vector2::new(-4500.0, 0.0));

        dirs.push(goaldir);
        for own_player in world.own_players.iter() {
            if own_player.id == player.id {
                continue;
            }
            dirs.push(Angle::between_points(player.position, own_player.position));
        }
        // find one that is closest to the current orientation
        // give priority to shooting into general enemy goals direction

        let mut target = Angle::from_radians(0.0);
        let mut min_badness = player.yaw.radians().abs();
        for dir in dirs {
            let mut badness = (dir - player.yaw).radians().abs();
            badness = badness - (dir - our_goaldir).radians().abs();
            if badness < min_badness {
                min_badness = badness;
                target = dir;
            }
        }
        target
    }
}

impl Role for FreeAttacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let player_data = ctx.player;
        let world_data = ctx.world;

        let (ball, init_ball) = if let Some(ball) = &world_data.ball {
            (ball, self.init_ball.get_or_insert(ball.clone()))
        } else {
            return PlayerControlInput::new();
        };
        let ball_pos = self.init_ball.as_ref().unwrap().position.xy();

        let target = if let Some(target) = self.target_direction {
            target
        } else {
            let target = self.find_best_direction(ball_pos, player_data, world_data);
            self.target_direction = Some(target);
            target
        };

        skill!(ctx, FetchBallWithHeading::new(target));

        skill!(ctx, ApproachBall::new());
        skill!(ctx, Kick::new());

        PlayerControlInput::new()
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
    fn update(&mut self, ctx: StrategyCtx) {
        let world = ctx.world;
        let us_attacking = world.current_game_state.us_operating;

        // Assign roles to players
        for player_data in world.own_players.iter() {
            if let Some(gate_keeper_id) = self.gate_keeper_id {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(e) = self.roles.entry(player_data.id) {
                if us_attacking {
                    if !self.has_attacker {
                        log::info!("Attacker is created");
                        self.has_attacker = true;
                        e.insert(Box::new(FreeAttacker::new()));
                    }
                } else if let Some(ball) = &world.ball {
                    let ball_pos = ball.position;
                    // get the disance between the ball and the player
                    let distance = (player_data.position - ball_pos.xy()).norm();
                    if distance < 500.0 {
                        log::info!("Player {} is moving out of the ball", player_data.id);
                        // get the target pos that is 500.0 away from the ball
                        let target = ball_pos.xy()
                            + (player_data.position - ball_pos.xy()).normalize() * 500.0;
                        e.insert(Box::new(OtherPlayer::new(target)));
                    }
                }
            }
        }

        // let mut inputs = PlayerInputs::new();
        // for (id, role) in self.roles.iter_mut() {
        //     if let Some(player_data) = world.own_players.iter().find(|p| p.id == *id) {
        //         let player_data = player_data.clone();
        //         let mut input = role.update(RoleCtx::new(&player_data, world, &mut HashMap::new()));
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
