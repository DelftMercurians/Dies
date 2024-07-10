use dies_core::Angle;
use dies_core::BallData;
use nalgebra::Vector2;

use dies_core::FieldGeometry;

use crate::{
    roles::Role,
    PlayerControlInput,
};

pub struct Harasser {
    distance_behind_ball: f64,
}
use super::RoleCtx;

impl Harasser {
    pub fn new(distance_behind_ball: f64) -> Self {
        Self { distance_behind_ball }
    }

    fn find_intersection(&self, ball: &BallData, world: &FieldGeometry) -> Vector2<f64> {
        let ball_pos = ball.position.xy();
        let half_length = world.field_length / 2.0;
        let goal_center = Vector2::new(-half_length, 0.0);

        // Compute the direction vector from the ball to the goal center
        let direction = (goal_center - ball_pos).normalize();

        let target_pos = ball_pos + direction * self.distance_behind_ball;
        
        // Ensuring the target position is infront of the line between the ball and the goal center
        if (target_pos - goal_center).dot(&direction) < 0.0 {
            return target_pos;
        }

        target_pos
    }

}

impl Role for Harasser {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let target_pos = self.find_intersection(ball, geom);
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
