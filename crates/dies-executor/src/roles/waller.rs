use dies_core::Angle;
use dies_core::BallData;
use dies_core::PlayerData;
use nalgebra::Vector2;

use crate::{
    roles::{skills::GoToPositionSkill, Role},
    skill, PlayerControlInput,
};

pub struct Waller {
    offset: f64,
}
use super::RoleCtx;

impl Waller {
    pub fn new(offset: f64) -> Self {
        Self { offset }
    }

    fn find_intersection(&self, _player_data: &PlayerData, ball: &BallData) -> Vector2<f64> {
        let goal_center = Vector2::new(6117.0, -129.0); // Center of the goal area
        let ball_pos = ball.position.xy();

        // Compute the direction vector from the ball to the goal center
        let direction: Vector2<f64> = goal_center - ball_pos;
        // Normalize the direction vector
        let direction = direction.normalize();
        println!("Direction: {:?}", direction);

        // Points representing the boundary of the goalkeeper area
        let area_top = 1400.0; // top boundary y-coordinate
        let area_bottom = -1400.0; // bottom boundary y-coordinate
        let area_right = 4630.0; // right boundary x-coordinate

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

        // fn find_intersection(&self, player_data: &PlayerData, ball: &BallData) -> Vector2<f64> {
        //     let goal_center = Vector2::new(6117.0, -129.0); // Center of the goal area
        //     let ball_pos = Vector2::new(ball.position.x, ball.position.y);

        //     // Compute the direction vector from the ball to the goal center
        //     let direction: Vector2<f64> = goal_center - ball_pos;
        //     // Normalize the direction vector
        //     let direction = direction.normalize();
        //     println!("Direction: {:?}", direction);

        //     // Points representing the boundary of the goalkeeper area
        //     let area_top = 1400.0; // top boundary y-coordinate
        //     let area_bottom = -1400.0; // bottom boundary y-coordinate
        //     let area_right = 4630.0;  // right boundary x-coordinate

        //     // Intersect with the top boundary
        //     if direction.y != 0.0 {

        //         let t_top = (area_top - ball_pos.y) / direction.y;
        //         let x_top = ball_pos.x + t_top * direction.x;

        //         let t_bottom = (area_bottom - ball_pos.y) / direction.y;
        //         let x_bottom = ball_pos.x + t_bottom * direction.x;

        //         if player_data.position.y < area_top && player_data.position.y > area_bottom {
        //             if x_bottom.abs() >= area_right && x_bottom.abs() <= 6125.0 {
        //                 let bottom_corner = Vector2::new(area_right, area_bottom);
        //                 if player_data.position != bottom_corner {
        //                     return bottom_corner;
        //                 } else {
        //                     return Vector2::new(x_bottom, area_bottom);
        //                 }
        //             }
        //             if x_top.abs() >= area_right && x_top.abs() <= 6125.0 {
        //                 // First go to the top corner
        //                 let top_corner = Vector2::new(area_right, area_top);
        //                 println!("Moving to top corner: {:?}", top_corner);
        //                 if player_data.position != top_corner {
        //                     return top_corner;
        //                 } else {
        //                     return Vector2::new(x_top, area_top);
        //                 }
        //             }
        //         }

        //         // Normal intersection checks
        //         if x_top.abs() >= area_right && x_top.abs() <= 6125.0 {
        //             return Vector2::new(x_top, area_top);
        //         }

        //         if x_bottom.abs() >= area_right && x_bottom.abs() <= 6125.0 {
        //             return Vector2::new(x_bottom, area_bottom);
        //         }
        //         }

        //         if direction.x != 0.0 {
        //             let t = (area_right - ball_pos.x) / direction.x;
        //             let y = ball_pos.y + t * direction.y;
        //             if y.abs() <= area_top {
        //                 return Vector2::new(area_right, y);
        //             }
        //         }

        // Default fallback to ball position (should not happen in normal cases)
        // CHANGE THIS TO MIDDLE OF THE GOAL
        //println!("Falling back to ball position");
        // return Vector2::new(area_right, ball_pos.y)
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
        let mut input = PlayerControlInput::new();
        input.with_position(target_pos);
        input.with_yaw(Angle::from_radians(
            self.angle_to_ball(ctx.player, ball_pos),
        ));
        input
    }
}
