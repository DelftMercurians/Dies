use dies_core::BallData;
use dies_core::{PlayerData, WorldData};
use nalgebra::Vector2;

use crate::strategy::Role;
use crate::PlayerControlInput;

pub struct Waller {}

impl Waller {
    pub fn new() -> Self {
        Self {}
    }

    // HAVE A LOOK AT THE CORRECT DIMENSIONS OF THE GOALKEEPER AREA ONCE YOU CAN RUN THE CODE!
    fn find_intersection(&self, player_data: &PlayerData, ball: &BallData, goal_width: f64) -> Vector2<f64> {
        let goal_center = Vector2::new(0.0, 0.0); // Center of the goal area
        let ball_pos = ball.position;

        // Compute the direction vector from the ball to the goal center
        let direction = goal_center - ball_pos;

        // Normalize the direction vector
        let direction = direction.normalize();

        // Points representing the boundary of the goalkeeper area
        let area_top = 1000.0; // top boundary y-coordinate
        let area_bottom = -1000.0; // bottom boundary y-coordinate
        let area_right = 1000.0;  // right boundary x-coordinate

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
    fn update(&mut self, player_data: &PlayerData, world: &WorldData) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();

        if let (Some(ball), Some(field_geom)) = (world.ball.as_ref(), world.field_geom.as_ref()) {
            let ball_y = ball.position.y;
            let ball_vy = ball.velocity.y;
            let player_y = player_data.position.y;

            if (player_y < ball_y && ball_vy < 0.0) || (player_y > ball_y && ball_vy > 0.0) {
                let target_pos: nalgebra::Matrix<
                    f64,
                    nalgebra::Const<2>,
                    nalgebra::Const<1>,
                    nalgebra::ArrayStorage<f64, 2, 1>,
                >;
                target_pos =
                    self.find_intersection(player_data, ball, field_geom.goal_width as f64);

                // let target_pos: nalgebra::Matrix<f64, nalgebra::Const<2>, nalgebra::Const<1>, nalgebra::ArrayStorage<f64, 2, 1>> = self.find_intersection(_player_data, _world);
                let target_angle = self.angle_to_ball(player_data, ball);

                input.with_position(target_pos);
                input.with_orientation(target_angle);
            }
        }

        input
    }
}
