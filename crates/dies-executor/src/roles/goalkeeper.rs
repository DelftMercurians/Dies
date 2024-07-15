use dies_core::{Angle, BallData};
use dies_core::{PlayerData};
use nalgebra::Vector2;

use crate::roles::{Role, RoleCtx};
use crate::PlayerControlInput;


pub struct Goalkeeper {}

impl Goalkeeper {
    pub fn new() -> Self {
        Self {}
    }

    fn find_intersection(
        &self,
        player_data: &PlayerData,
        ball: &BallData,
        goal_width: f64,
    ) -> Vector2<f64> {
        // Find the goalkeeper's position
        let goalkeeper_pos = player_data.position;

        // Get the ball's current position and velocity
        let ball_pos = ball.position;
        let ball_vel = ball.velocity;

        // Calculate the second point based on the ball's trajectory
        let second_point = ball_pos - ball_vel;

        // Coordinates of the ball's trajectory points
        let x1 = ball_pos.x;
        let y1 = ball_pos.y;
        let x2 = second_point.x;
        let y2 = second_point.y;

        // Vertical line's x-coordinate is the goalkeeper's x-coordinate
        // Maybe it would be better to get a static value so that it doesn't deviate over time...
        let x_vertical = 0.0; // goalkeeper_pos.x;

        // Handle the special case where the ball's trajectory is vertical
        if x1 == x2 {
            return Vector2::new(x_vertical, ball_pos.y);
        }

        // Calculate the parameter t to find the y-coordinate at the intersection with the vertical line
        let t = (x_vertical - x1) / (x2 - x1);
        let py = y1 + t * (y2 - y1);

        // Calculate the min and max y-coordinates the goalkeeper can move to, constrained by the goal width
        let y_min = goalkeeper_pos.y - goal_width / 2.0;
        let y_max = goalkeeper_pos.y + goal_width / 2.0;

        // Clamp the y-coordinate to ensure it stays within the goal width range
        let py_clamped = py.max(y_min).min(y_max);

        println!("{}, {}", x_vertical, py_clamped);

        // Return the clamped intersection point
        Vector2::new(x_vertical, py_clamped)
    }
}

impl Role for Goalkeeper {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        let mut input = PlayerControlInput::new();
        let world = ctx.world;
        let player_data = ctx.player;

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
                > =
                    self.find_intersection(player_data, ball, field_geom.goal_width);

                let target_angle = Angle::between_points(player_data.position, ball.position.xy());

                input.with_position(target_pos);
                input.with_yaw(target_angle);
            }
        }

        input
    }
}
