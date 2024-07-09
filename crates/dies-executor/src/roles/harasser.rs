use dies_core::Angle;
use dies_core::BallData;
use dies_core::PlayerData;
use nalgebra::Vector2;

use dies_core::FieldGeometry;

use crate::{
    roles::Role,
    PlayerControlInput,
};

pub struct Harasser {
    pub distance_behind_ball: f64,
}
use super::RoleCtx;

impl Harasser {
    pub fn new(distance_behind_ball: f64) -> Self {
        Self { distance_behind_ball }
    }

    fn find_intersection(&self, _player_data: &PlayerData, ball: &BallData, _world: &FieldGeometry) -> Vector2<f64> {
        let goal_center = Vector2::new(6117.0, -129.0); // Center of the goal area
        // let goal_center = Vector2::new(world.field_length / 2, 0);
        // let goal_center_f64 = goal_center.map(|e| e as f64);
        let ball_pos = ball.position.xy();

        // Compute the direction vector from the ball to the goal center
        let direction: Vector2<f64> = goal_center - ball_pos;
        // Normalize the direction vector
        let direction = direction.normalize();
        //dies_core::debug_line(ball_pos, ball_pos + direction, dies_core::DebugColor::Red);

        let target_pos = ball_pos + direction * self.distance_behind_ball;
        
        // Ensuring the target position is infront of the line between the ball and the goal center
        if (target_pos - goal_center).dot(&direction) < 0.0 {
            return target_pos;
        }

        target_pos
    }

    fn angle_to_ball(&self, player_data: &PlayerData, ball: &BallData) -> f64 {
        // Angle needed to face the passer (using arc tangent)
        let receiver_pos = player_data.position;
        let dx = ball.position.x - receiver_pos.x;
        let dy = ball.position.y - receiver_pos.y;

        dy.atan2(dx)
    }
}

impl Role for Harasser {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let ball_pos = ctx.world.ball.as_ref().unwrap();
        let world = ctx.world.field_geom.as_ref().unwrap();
        let target_pos = self.find_intersection(ctx.player, ball_pos, world);
        let mut input = PlayerControlInput::new();
        input.with_position(target_pos);
        input.with_yaw(Angle::from_radians(
            self.angle_to_ball(ctx.player, ball_pos),
        ));
        input
    }
}
