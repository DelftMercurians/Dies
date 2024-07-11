use dies_core::{Angle, BallData, FieldGeometry, Vector2, WorldData};

use super::RoleCtx;
use crate::{roles::Role, PlayerControlInput};

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Attacker {
    position: Vec<Vector2>,
}

impl Attacker {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(position: Vec<Vector2>) -> Self {
        Self { position }
    }

    /// Calculate the distance between two points.
    fn distance(a: Vector2, b: Vector2) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
    }

    fn aim_at_goal(player_pos: Vec<Vector2>, world: &WorldData) -> Angle {
        let opp_goal_center = - world.field_length / 2.0;
        let angle = Angle::between_points(player_pos, opp_goal_center);
        return angle;
    }

    fn aim_at_ball(player_pos: Vec<Vector2>, ball_pos: Vec<Vector2>) -> Angle {
        let angle = Angle::between_points(player_pos, ball_pos);
        return angle;
    }

    fn closest_player_to_ball(world: &WorldData) -> Option<PlayerData> {
        let mut min_distance = f64::MAX;
        let mut closest_player = None;
        for player in world.own_players.iter() {
            let distance = distance(player.position, world.ball.position.xy());
            if distance < min_distance {
                min_distance = distance;
                closest_player = Some(player);
            }
        }
        return closest_player;
    }

    /// Go to given position
    fn go_to_pos() -> Vector2 {
        return self.position;
    }

    fn closest_player_to_passer(world: &WorldData, passer_pos: Vec<Vector2>) -> Option<u8> {
        let mut min_distance = f64::MAX;
        let mut closest_player = None;
        for player in world.own_players.iter() {
            let distance = distance(player.position, passer_pos);
            if distance < min_distance {
                min_distance = distance;
                closest_player = Some(player.id);
            }
        }
        return closest_player;
    
    }
}

impl Role for Attacker {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let passer = closest_player_to_ball(ctx.world);
            for player in ctx.players.iter() {
                if player.id == passer.id {
                    let target_pos = ball.position.xy();
                    let target_angle = aim_at_ball(ctx.player.position, target_pos);
                    let dribble_speed = 1.0;
                    let mut input = PlayerControlInput::new();
                    while player.breakbeam_ball_detected == false {
                        input.with_position(target_pos);
                        input.with_yaw(target_angle);
                        input.with_dribbling_speed(dribble_speed);
                        return input;
                    }
                }
                input.with_position(self.position);
                input.with_yaw(aim_at_goal(ctx.player.position, ctx.world));
            }
            let target_pos = self.go_to_pos();
            let mut input = PlayerControlInput::new();
            input.with_position(target_pos);
            input.with_yaw(Angle::between_points(
                ctx.player.position,
                ball.position.xy(),
            ));
            input
        } else {
            PlayerControlInput::new()
        }
    }
}
