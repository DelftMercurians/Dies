//! Goalkeeper — dedicated positioning, outside Formation, never the active robot.
//!
//! Sits on a line just in front of the goal and tracks the ball's projection onto
//! the goal mouth, covering the shot angle.

use dies_strategy_api::prelude::*;
use dies_strategy_api::World;

/// Compute the keeper's target position: on the keeper line, at the point where
/// the ball→goal-centre ray crosses it, clamped to the goal mouth.
pub fn keeper_target(world: &World, depth: f64) -> Vector2 {
    let own_goal = world.own_goal_center();
    let keeper_x = own_goal.x + depth; // own_goal.x is negative; move toward field
    let half_goal = world.goal_width() / 2.0;

    let y = match world.ball_position() {
        Some(ball) => {
            let dx = own_goal.x - ball.x;
            if dx.abs() < 1e-6 {
                ball.y
            } else {
                let t = (keeper_x - ball.x) / dx;
                ball.y + t * (own_goal.y - ball.y)
            }
        }
        None => 0.0,
    };

    Vector2::new(keeper_x, y.clamp(-half_goal, half_goal))
}
