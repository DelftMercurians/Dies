use crate::roles::skills::{ApproachBall, FetchBallWithHeading, Kick};
use crate::roles::{Goalkeeper, RoleCtx, SkillResult};
use crate::strategy::kickoff::OtherPlayer;
use crate::strategy::{Role, Strategy};
use crate::{skill, PlayerControlInput};
use dies_core::{Angle, BallData, PlayerData, PlayerId, RoleType, Vector2, FieldGeometry, WorldData};
use nalgebra::{Rotation2, SMatrix, SquareMatrix};
// use num_traits::float::Float;
use std::collections::HashMap;
use std::fmt::format;

use super::StrategyCtx;

pub struct FreeKickStrategy {
    roles: HashMap<PlayerId, Box<dyn Role>>,
    kicker_id: Option<PlayerId>,
    gate_keeper_id: Option<PlayerId>,
}

pub struct FreeAttacker {
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
        Angle::between_points(player.position, Vector2::new(4500.0, 100.0 * player.position.y.signum()))

        // dirs.push(goaldir);
        // for own_player in world.own_players.iter() {
        //     if own_player.id == player.id {
        //         continue;
        //     }
        //     dirs.push(Angle::between_points(player.position, own_player.position));
        // }
        // // find one that is closest to the current orientation
        // // give priority to shooting into general enemy goals direction

        // let mut target = Angle::from_radians(0.0);
        // let mut min_badness = player.yaw.radians().abs();
        // for dir in dirs {
        //     let mut badness = (dir - player.yaw).radians().abs();
        //     badness = badness - (dir - our_goaldir).radians().abs();
        //     if badness < min_badness {
        //         min_badness = badness;
        //         target = dir;
        //     }
        // }
        // target
    }
}

impl Role for FreeAttacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let player_data = ctx.player;
        let world_data = ctx.world;

        let (_ball, init_ball) = if let Some(ball) = &world_data.ball {
            (ball, self.init_ball.get_or_insert(ball.clone()))
        } else {
            return PlayerControlInput::new();
        };
        let ball_pos = init_ball.position.xy();

        let target = if let Some(target) = self.target_direction {
            target
        } else {
            let target = self.find_best_direction(ball_pos, player_data, world_data);
            self.target_direction = Some(target);
            target
        };

        skill!(ctx, FetchBallWithHeading::new(target));

        loop {
            skill!(ctx, ApproachBall::new());
            if let SkillResult::Success = skill!(ctx, Kick::new()) {
                break;
            }
        }

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
            kicker_id: None,
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
    fn name(&self) -> &'static str {
        "FreeKick"
    }

    fn on_enter(&mut self, _ctx: StrategyCtx) {
        // Clear roles
        self.roles.clear();
        self.kicker_id = None;
    }

    fn update(&mut self, ctx: StrategyCtx) {
        let world = ctx.world;
        let us_attacking = world.current_game_state.us_operating;
        
        if let (Some(ball), Some(field)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let kicker_id = if us_attacking {
                Some(*self.kicker_id.get_or_insert_with(|| {
                    let kicker_id = world
                        .own_players
                        .iter()
                        .filter(|p| Some(p.id) != self.gate_keeper_id)
                        .min_by_key(|p| (ball.position.xy() - p.position).norm() as i64)
                        .unwrap()
                        .id;
                    self.roles.insert(
                        kicker_id,
                        Box::new(FreeAttacker {
                            init_ball: None,
                            target_direction: None,
                        }),
                    );
                    kicker_id
                }))
            } else {
                None
            };

            // Assign roles to players
            for player_data in world.own_players.iter() {
                if kicker_id == Some(player_data.id) && !self.roles.contains_key(&player_data.id) {
                    self.roles.insert(
                        player_data.id,
                        Box::new(Goalkeeper::new())
                    );
                    continue;
                }
                if self.gate_keeper_id == Some(player_data.id) || kicker_id == Some(player_data.id)
                {
                    continue;
                }

                if let std::collections::hash_map::Entry::Vacant(e) =
                    self.roles.entry(player_data.id)
                {
                    let ball_pos = ball.position;
                    // get the disance between the ball and the player
                    let distance = (player_data.position - ball_pos.xy()).norm();
                    if distance < 650.0 {
                        // get the target pos that is 500.0 away from the ball
                        // in the same direction between the robot and the ball
                        // and resampling further if they have a collision with other players                        

                        let mut target = ball_pos.xy()
                            + (player_data.position - ball_pos.xy()).normalize() * 650.0;
                        
                        // create a vector with the positions of all the players from the ctx
                        // ctx.world.own_players

                        // let player_diameter = 90.0;
                        // let mut extra_spacing = player_diameter;

                        // let mut collision = true;
                        // let mut all_players = ctx.world.own_players.clone();
                        // let mut opp_players = ctx.world.opp_players.clone();



                        // // let all_players = own_players.append(opp_players.as_mut());
                        // all_players.append(opp_players.as_mut());

                        // // create a big list of different positions at different angles
                        // let spacings = vec![extra_spacing * 0.0, extra_spacing * 1.0, extra_spacing * 2.0, extra_spacing * 3.0];

                        // // then a list of angles in degrees
                        // let angles = vec![0.0, 30.0, -30.0, 90.0, -90.0];

                        // for ang in angles.iter() {
                        //     for space in spacings.iter() {
                        //         // compute the target with this angle and spacing

                        //         // rotation matrix
                        //         // let rot_mat = SMatrix::<f64, 2, 2>::new(ang.cos() as f64, -ang.sin() as f64, ang.sin() as f64, ang.cos() as f64);
                        //         // let rot_mat = SquareMatrix::new()
                        //         // let rot_matrix = Rotation2::new(ang);
                                
                        //         let mut original_direction = (player_data.position - ball_pos.xy()).normalize().clone();
                                
                        //         let angle_t = Angle::from_degrees(*ang) - Angle::from_vector(original_direction);

                        //         target = ball_pos.xy() + angle_t.rotate_vector(&original_direction) * (650.0 + space);

                        //         // with this specific configuration then check for all the players
                        //         collision = false;
                        //         for other_player in all_players.iter() {
                        //             if other_player.id == player_data.id {
                        //                 continue;
                        //             }
    
                        //             let distance = (other_player.position - target).norm();
                                    
                        //             if distance < player_diameter {
                        //                 // then this target is not valid
                        //                 // skip and test the next one

                        //                 collision = true;
                        //                 break;
                        //             }
                        //         }

                        //         if !collision {
                        //             // if no collision was found, we have a good target
                        //             break; 
                        //         }
                        //     }
                        // }

                        // if (target.y - world.field_geom.as_ref().unwrap().field_width / 2.0).abs()
                        //     < 600.0
                        // {
                        //     target.y = ball_pos.y + 650.0;
                        // }
                        // if (target.x - world.field_geom.as_ref().unwrap().field_length / 2.0).abs()
                        //     < 600.0
                        // {
                        //     target.x = ball_pos.x + 650.0;
                        // }

                        let min_theta = -120;
                        let max_theta = 120;
                        let max_radius = 1000;
                        let min_distance = 700.0;
                        let original_direction = (player_data.position - ball_pos.xy()).normalize().clone();
                        let mut i = 0;
                        target = nearest_safe_pos(ball_pos.xy(), min_distance, player_data.position, min_theta, max_theta, max_radius, field);
                        dies_core::debug_line(format!("p{}.line_ball", player_data.id), Vector2::new(ball_pos.x, ball_pos.y), player_data.position, dies_core::DebugColor::Red);
                        
                        dies_core::debug_string(format!("p{}.FinalTarget", player_data.id), target.to_string());
                        dies_core::debug_line(format!("p{}.line_final_target", player_data.id), Vector2::new(ball_pos.x, ball_pos.y), target, dies_core::DebugColor::Orange);

                        // for theta in (min_theta as i32..max_theta as i32).step_by(10) {
                        //     let theta = (theta as f64).to_radians();
                        //     let theta_line = Angle::from_vector(original_direction).radians();
                            
                        //     for radius in (0..max_radius as i32).step_by(20) {

                        //         let x = ball_pos.x + (min_distance + radius as f64) * (theta + theta_line).cos();
                        //         let y = ball_pos.y + (min_distance + radius as f64) * (theta + theta_line).sin();
                        //         let position = Vector2::new(x, y);
                                
                        //         i = i + 1;
                        //         if is_pos_in_field(position, field) {
                        //             // dies_core::debug_cross(format!("p{}.CheckingPos{}",player_data.id , i), position, dies_core::DebugColor::Purple);
                        //             // dies_core::debug_string(format!("p{}.FinalTarget", player_data.id), position.to_string());
                        //             // dies_core::debug_line(format!("p{}.line_final_target", player_data.id), Vector2::new(ball_pos.x, ball_pos.y), target, dies_core::DebugColor::Orange);

                        //             target = position;
                        //             break;
                        //         } else {
                        //             dies_core::debug_cross(format!("p{}.CheckingPos{}",player_data.id , i), position, dies_core::DebugColor::Orange);
                        //         }
                        //     }
                        // }
                        // dies_core::debug_string("field.length", field.field_length.to_string());
                        // dies_core::debug_string("field.width", field.field_width.to_string());
                        e.insert(Box::new(OtherPlayer::new(target)));
                    }
                }
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

fn nearest_safe_pos(avoding_point : Vector2, min_distance : f64, initial_pos: Vector2, min_theta :i32, max_theta:i32, max_radius: i32, field: &FieldGeometry) -> Vector2 {
    let mut i = 0;
    let mut target = Vector2::new(0.0, 0.0);

    // create a vector between avoiding_point and initial_pos
    let original_direction = (initial_pos - avoding_point).normalize().clone();

    for theta in (min_theta as i32..max_theta as i32).step_by(10) {
        let theta = (theta as f64).to_radians();
        let theta_line = Angle::from_vector(original_direction).radians();
        
        for radius in (0..max_radius as i32).step_by(20) {

            let x = avoding_point.x + (min_distance + radius as f64) * (theta + theta_line).cos();
            let y = avoding_point.y + (min_distance + radius as f64) * (theta + theta_line).sin();
            let position = Vector2::new(x, y);
            
            i = i + 1;
            if is_pos_in_field(position, field) {
                // dies_core::debug_cross(format!("p{}.CheckingPos{}",player_data.id , i), position, dies_core::DebugColor::Purple);
                // dies_core::debug_string(format!("p{}.FinalTarget", player_data.id), position.to_string());
                // dies_core::debug_line(format!("p{}.line_final_target", player_data.id), Vector2::new(ball_pos.x, ball_pos.y), target, dies_core::DebugColor::Orange);

                return position;
            } 
        }
    }
    target
}

fn is_pos_in_field(pos: Vector2, field: &FieldGeometry) -> bool {
    const MARGIN: f64 = 100.0;
    // check if pos outside field
    if pos.x.abs() > field.field_length / 2.0  - MARGIN
        || pos.y.abs() > field.field_width / 2.0 - MARGIN
    {
        return false;
    }
    
    true
}
