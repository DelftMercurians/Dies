use dies_core::{Angle, FieldGeometry, Vector2, WorldData, PlayerData, PlayerId};
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

    fn aim_at_ball(&self, player_pos: Vector2, ball_pos: Vector2) -> Angle {
        let angle = Angle::between_points(player_pos, ball_pos);
        return angle;
    }

    fn closest_player_to_ball<'a>(&self, world: &'a WorldData) -> Option<&'a PlayerData> {
        let ball = world.ball.as_ref().unwrap();
        let mut min_distance = f64::MAX;
        let mut closest_player = None;
        for player in world.own_players.iter() {
            let distance = self.distance(player.position, ball.position.xy());
            if distance < min_distance {
                min_distance = distance;
                closest_player = Some(player);
            }
        }
        return closest_player;
    }

    /// Go to given position
    fn go_to_pos(&self) -> Vector2 {
        return self.position;
    }

    fn closest_player_to_passer(&self, world: &WorldData, passer_pos: Vector2) -> Option<PlayerId> {
        let mut min_distance = f64::MAX;
        let mut closest_player = None;
        for player in world.own_players.iter() {
            let distance = self.distance(player.position, passer_pos);
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
            let passer = self.closest_player_to_ball(ctx.world);
            let mut input = PlayerControlInput::new();
            for player in ctx.world.own_players.iter() {
                if player.id == passer.unwrap().id {
                    let target_pos = ball.position.xy();
                    let target_angle = self.aim_at_ball(ctx.player.position, target_pos);
                    let dribble_speed = 10.0;
                    while self.distance(player.position, ball.position.xy()) > 300.0 {
                        // println!("{}", player.breakbeam_ball_detected);
                        input.with_position(target_pos);
                        input.with_yaw(target_angle);
                        input.with_dribbling(dribble_speed);
                        input.with_kicker(KickerControlInput::Arm);
                        return input;
                    }
                }
                if self.distance(player.position, self.position) < 100.0 {
                    input.with_kicker(KickerControlInput::Kick);
                    return input;
                }
                input.with_dribbling(10.0);
                input.with_position(self.position);
                input.with_yaw(self.aim_at_goal(ctx.player.position, geom));
                return input;

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
