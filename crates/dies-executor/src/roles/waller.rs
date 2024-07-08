use dies_core::BallData;
use dies_core::PlayerData;
use nalgebra::Vector2;

use crate::{
    roles::{skills::GoToPositionSkill, Role},
    skill, PlayerControlInput,
};


pub struct Waller {}
use super::RoleCtx;

impl Waller {
    pub fn new() -> Self {
        Self {}
    }

    // HAVE A LOOK AT THE CORRECT DIMENSIONS OF THE GOALKEEPER AREA ONCE YOU CAN RUN THE CODE!
    fn find_intersection(&self, player_data: &PlayerData, ball: &BallData) -> Vector2<f64> {
        let goal_center = Vector2::new(6117.0, -100.0); // Center of the goal area
        let ball_pos = Vector2::new(ball.position.x, ball.position.y);

        // Compute the direction vector from the ball to the goal center
        // let direction = goal_center - ball_pos;
        let direction: Vector2<f64> = goal_center - ball_pos;
        // Normalize the direction vector
        let direction = direction.normalize();

        // Points representing the boundary of the goalkeeper area
        let area_top = 1400.0; // top boundary y-coordinate
        let area_bottom = -1400.0; // bottom boundary y-coordinate
        let area_right = 4485.0;  // right boundary x-coordinate

        // Intersect with the top boundary
        if direction.y != 0.0 {
            let t = (area_top - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            if x.abs() <= area_right {
                return Vector2::new(x, area_top);
            }

            // Intersect with the bottom boundary
            let t = (area_bottom - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            if x.abs() <= area_right {
                return Vector2::new(x, area_bottom);
            }
        }

        // Intersect with the right boundary
        if direction.x != 0.0 {
            let t = (area_right - ball_pos.x) / direction.x;
            let y = ball_pos.y + t * direction.y;
            if y.abs() <= area_top {
                return Vector2::new(area_right, y);
            }
        }

        // Default fallback to ball position (should not happen in normal cases)
        // CHANGE THIS TO MIDDLE OF THE GOAL 
        return Vector2::new(area_right, ball_pos.y)
    }


    fn angle_to_ball(&self, player_data: &PlayerData, ball: &BallData) -> f64 {
        // Angle needed to face the passer (using arc tangent)
        let receiver_pos = player_data.position;
        let dx = ball.position.x - receiver_pos.x;
        let dy = ball.position.y - receiver_pos.y;

        dy.atan2(dx)
    }
}

impl Role for Waller {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let ball_pos = ctx.world.ball.as_ref().unwrap();
        let target_pos = self.find_intersection(ctx.player, ball_pos);
        skill!(ctx, GoToPositionSkill::new(target_pos));
        // input.with_orientation(target_angle);
        PlayerControlInput::new()
    }
}