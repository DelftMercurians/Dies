use dies_core::{BallData, FieldGeometry, PlayerId, Vector2};

use super::RoleCtx;
use crate::{roles::Role, PlayerControlInput};

const MARGIN: f64 = 200.0;

/// A role that moves the player to the intersection of the ball's path with the goal
/// line, acting as a wall to block the ball from reaching the goal.
pub struct Waller {
    offset: f64,
}

impl Waller {
    /// Create a new Waller role with the given offset from the intersection point.
    pub fn new(offset: f64) -> Self {
        Self { offset }
    }

    /// Find the intersection point of the ball's path with the goal line and return the
    /// target position for the player.
    fn find_intersection(
        &self,
        ball: &BallData,
        world: &FieldGeometry,
        player_id: PlayerId,
    ) -> Vector2 {
        let ball_pos = ball.position.xy();
        let half_length = world.field_length / 2.0;
        let goal_center = Vector2::new(-half_length, 0.0);

        // Compute the direction vector from the ball to the goal center
        let direction = (goal_center - ball_pos).normalize();

        // Points redebuesenting the boundary of the goalkeeper area
        let area_top_y = world.penalty_area_width / 2.0 + MARGIN; // top boundary y-coordinate
        let area_bottom_y = -area_top_y - MARGIN; // bottom boundary y-coordinate
        let area_right_x = -half_length + world.penalty_area_depth + MARGIN; // right boundary x-coordinate

        if direction.y != 0.0 {
            // Intersect with the bottom boundary
            let t = (area_bottom_y - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            if x <= area_right_x && x >= -half_length && ball_pos.y < 0.0 {
                return Vector2::new(x + self.offset, area_bottom_y);
            }

            // Intersect with the top boundary
            let t = (area_top_y - ball_pos.y) / direction.y;
            let x = ball_pos.x + t * direction.x;
            if x <= area_right_x && x >= -half_length && ball_pos.y > 0.0 {
                return Vector2::new(x + self.offset, area_top_y);
            }
        }

        // Intersect with the right boundary
        if direction.x != 0.0 {
            let t = (area_right_x - ball_pos.x) / direction.x;
            let mut y = ball_pos.y + t * direction.y;
            if y + self.offset > world.penalty_area_width / 2.0 {
                y = world.penalty_area_width / 2.0 - self.offset;
            }
            if y + self.offset < -world.penalty_area_width / 2.0 {
                y = -world.penalty_area_width / 2.0 - self.offset;
            }
            if y.abs() <= area_top_y {
                return Vector2::new(area_right_x, y + self.offset);
            }
        }

        // Default fallback to ball position (should not happen in normal cases)
        Vector2::new(area_right_x, ball_pos.y)
    }
}

impl Role for Waller {
    fn update(&mut self, ctx: RoleCtx<'_>) -> PlayerControlInput {
        if let (Some(ball), Some(geom)) = (ctx.world.ball.as_ref(), ctx.world.field_geom.as_ref()) {
            let target_pos = self.find_intersection(ball, geom, ctx.player.id);
            let mut input = PlayerControlInput::new();
            input.with_position(target_pos);
            // input.with_yaw(Angle::between_points(
            //     ctx.player.position,
            //     ball.position.xy(),
            // ));
            input
        } else {
            PlayerControlInput::new()
        }
    }
}
