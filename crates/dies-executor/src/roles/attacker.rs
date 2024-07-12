use dies_core::{Angle, FieldGeometry, Vector2, WorldData, PlayerData, BallData, PlayerId};
use super::RoleCtx;
use crate::{roles::Role, PlayerControlInput, KickerControlInput};

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Attacker {
    position: Vector2,
}

impl Attacker {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(position: Vector2) -> Self {
        Self { position }
    }

    /// Calculate the distance between two points.
    fn distance(&self, a: Vector2, b: Vector2) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }

    fn aim_at_goal(&self, player_pos: Vector2, world: &FieldGeometry) -> Angle {
        let opp_goal_center = Vector2::new(- world.field_length / 2.0, 0.0);
        let angle = Angle::between_points(player_pos, opp_goal_center);
        return angle;
    }

    fn aim_at_player(&self, passer_pos: Vector2, reciever_pos: Vector2) -> Angle {
        let angle = Angle::between_points(passer_pos, reciever_pos);
        return angle;
    }

    fn aim_at_ball(&self, player_pos: Vector2, ball_pos: Vector2) -> Angle {
        let angle = Angle::between_points(player_pos, ball_pos);
        return angle;
    }

    fn closest_player_to_ball(&self, world: &WorldData, ball: &BallData) -> Option<PlayerId> {
        let mut min_distance = f64::MAX;
        let mut closest_player_id = None;
        for player in world.own_players.iter() {
            let distance = self.distance(player.position, ball.position.xy());
            if distance < min_distance {
                min_distance = distance;
                closest_player_id = Some(player.id);
            }
        }
        return closest_player_id;
    }

    fn closest_player_to_passer(&self, world: &WorldData, passer_pos: Vector2, passer_id: PlayerId) -> Option<PlayerId> {
        let mut min_distance = f64::MAX;
        let mut closest_player_id = None;
        for player in world.own_players.iter() {
            if player.id == passer_id {
                continue;
            }
            let distance = self.distance(player.position, passer_pos);
            if distance < min_distance {
                min_distance = distance;
                closest_player_id = Some(player.id);
            }
        }
        return closest_player_id;
    
    }
}

impl Role for Attacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let passer_id = self.closest_player_to_ball(ctx.world, ball);
            let passer_pos = ctx.world.own_players.iter().find(|p| p.id == passer_id.unwrap()).unwrap().position;
            let shooter_id = self.closest_player_to_passer(ctx.world, passer_pos, passer_id.unwrap());
            let mut input = PlayerControlInput::new();
                if ctx.player.id == passer_id.unwrap() {
                    // println!("passer ID: {}", passer_id.unwrap());
                    // let mut input = PlayerControlInput::new();
                    let target_pos = ball.position.xy();
                    let target_angle = self.aim_at_ball(ctx.player.position, target_pos);
                    let dribble_speed = 1.0;
                    while self.distance(ctx.player.position, ball.position.xy()) > 300.0 {
                        // Input to move to the ball and dribble.
                        input.with_position(target_pos);
                        input.with_yaw(target_angle);
                        input.with_dribbling(dribble_speed);
                        input.with_kicker(KickerControlInput::Arm);
                        return input;
                    }
                    // Input to find the closest player, then face the shooter and pass the ball
                    if self.distance(ctx.player.position, self.position) < 100.0 {
                        let shooter_id = self.closest_player_to_passer(ctx.world, passer_pos, passer_id.unwrap());
                        let shooter_pos = ctx.world.own_players.iter().find(|p| p.id == shooter_id.unwrap()).unwrap().position;
                        dies_core::debug_cross("ShooterPos", shooter_pos, dies_core::DebugColor::Purple);
                        let target_angle = self.aim_at_player(passer_pos, shooter_pos);
                        input.with_yaw(target_angle);
                        input.with_dribbling(1.0);
                        if (target_angle - ctx.player.yaw).abs() < 0.1 {
                            println!("Passing the ball");
                            input.with_kicker(KickerControlInput::Kick);
                        }
                        return input;
                        
                    }
                }

                if ctx.player.id == shooter_id.unwrap() {
                    let target_pos = self.position;
                    let target_angle = self.aim_at_player(passer_pos, ctx.player.position);
                    let dribble_speed = 1.0;
                    while self.distance(ctx.player.position, self.position) > 100.0 {
                        // Input to move to the ball and dribble.
                        input.with_position(target_pos);
                        input.with_yaw(target_angle);
                        input.with_dribbling(dribble_speed);
                        return input;
                    }

                    if self.distance(ctx.player.position, self.position) < 100.0 && self.distance(ctx.player.position, ball.position.xy()) < 300.0 {
                        let target_angle = self.aim_at_goal(ctx.player.position, geom);
                        input.with_yaw(target_angle);
                        input.with_dribbling(1.0);
                        input.with_kicker(KickerControlInput::Kick);
                        return input;
                    }
            }

                // If the shooter has the ball nearby he shoots
                
            // Input to move to the specified position and aim at the goal.
            input.with_dribbling(1.0);
            input.with_position(self.position);
            input.with_yaw(self.aim_at_goal(ctx.player.position, geom));
            return input;
            
        } else {
            PlayerControlInput::new()
        }
    }
}
