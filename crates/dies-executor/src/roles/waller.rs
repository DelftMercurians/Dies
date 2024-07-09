use dies_core::Angle;
use dies_core::BallData;
use dies_core::PlayerData;
use nalgebra::Vector2;
use nalgebra::Matrix;
use dies_core::FieldGeometry;

use crate::{
    roles::Role,
    PlayerControlInput,
};

pub struct Waller {
    pub offset: f64,
}
use super::RoleCtx;

impl Waller {
    pub fn new(offset: f64) -> Self {
        Self { offset }
    }

    fn find_intersection(&self, _player_data: &PlayerData, ball: &BallData, world: &FieldGeometry) -> Vector2<f64> {
        // let goal_center = Vector2::new(6117.0, -129.0); // Center of the goal area
        let goal_center = Vector2::new(world.field_length / 2, 0);
        let goal_center_f64 = goal_center.map(|e| e as f64);
        let ball_pos = ball.position.xy();

        // Compute the direction vector from the ball to the goal center
        let direction: Vector2<f64> = goal_center_f64 - ball_pos;
        // Normalize the direction vector
        let direction = direction.normalize();
        //dies_core::debug_line(ball_pos, ball_pos + direction, dies_core::DebugColor::Red);

        // Points representing the boundary of the goalkeeper area
        let area_top = world.goal_width as f64 / 2.0; // top boundary y-coordinate
        // let area_top = 1400.0; // top boundary y-coordinate
        let area_bottom = -area_top; // bottom boundary y-coordinate
        // let area_bottom = -1400.0; // bottom boundary y-coordinate
        let area_right = world.field_length as f64 / 2.0 - world.goal_depth as f64; // right boundary x-coordinate
        // let area_right = 4630.0; // right boundary x-coordinate

        // Intersect with the top boundary
        if direction.y != 0.0 {
            let t = (area_top - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            println!("Intersecting with top boundary: {:?}", x);
            if x >= area_right && x <= 6125.0 {
                return Vector2::new(x + self.offset, area_top);
            }

            // Intersect with the bottom boundary
            let t = (area_bottom - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            println!("Intersecting with bottom boundary: {:?}", x);
            if x >= area_right && x <= 6125.0 {
                return Vector2::new(x + self.offset, area_bottom);
            }
        }

        // Intersect with the right boundary
        if direction.x != 0.0 {
            let t = (area_right - ball_pos.x) / direction.x;
            let y = ball_pos.y + t * direction.y;
            if y.abs() <= area_top {
                return Vector2::new(area_right, y + self.offset);
            }
        }

        // Default fallback to ball position (should not happen in normal cases)
        // CHANGE THIS TO MIDDLE OF THE GOAL
        //println!("Falling back to ball position");
        return Vector2::new(area_right, ball_pos.y);
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
