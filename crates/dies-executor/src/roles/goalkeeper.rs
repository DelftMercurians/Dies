use dies_core::{find_intersection, Angle, BallData, PlayerData};
use nalgebra::Vector2;

use crate::roles::{Role, RoleCtx};
use crate::PlayerControlInput;

const KEEPER_X_OFFSET: f64 = 350.0;

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

        if let (Some(ball), Some(field_geom)) =
            (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref())
        {
            let ball_pos = ball.position.xy();
            let goalkeeper_x = -field_geom.field_length / 2.0 + KEEPER_X_OFFSET;
            // if (player_y < ball_y && ball_vy < 0.0) || (player_y > ball_y && ball_vy > 0.0) {

            let target_angle = Angle::between_points(ctx.player.position, ball_pos);

            input.with_yaw(target_angle);

            let defence_width = 1.3 * (field_geom.penalty_area_width / 2.0);
            let mut target = if ball.velocity.xy().norm() > 100.0 {
                find_intersection(
                    Vector2::new(goalkeeper_x, 0.0),
                    Vector2::y(),
                    ball_pos,
                    ball.velocity.xy(),
                )
                .unwrap_or(Vector2::new(goalkeeper_x, ball.position.y))
            } else {
                Vector2::new(goalkeeper_x, ball.position.y)
            };
            target.y = target.y.max(-defence_width).min(defence_width);
            input.with_position(target);
        }

        input
    }

    fn role_type(&self) -> dies_core::RoleType {
        dies_core::RoleType::Goalkeeper
    }
}
