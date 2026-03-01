use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn calculate_intercept_position(s: &RobotSituation) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        if ball.velocity.xy().norm() < 100.0 {
            return ball.position.xy();
        }

        let ball_pos = ball.position.xy();
        let ball_vel = ball.velocity.xy();
        let player_pos = s.player_data().position;

        // Simple intercept calculation
        let intercept_time = estimate_intercept_time(ball_pos, ball_vel, player_pos);
        let predicted_pos = ball_pos + ball_vel * intercept_time;

        predicted_pos
    } else {
        s.player_data().position
    }
}

fn estimate_intercept_time(ball_pos: Vector2, ball_vel: Vector2, player_pos: Vector2) -> f64 {
    let max_robot_speed = 3000.0; // mm/s

    // Iterative approximation
    let mut time_estimate = 0.5;
    for _ in 0..5 {
        let predicted_ball = ball_pos + ball_vel * time_estimate;
        let distance = (predicted_ball - player_pos).norm();
        time_estimate = distance / max_robot_speed;
    }

    time_estimate.min(3.0) // Cap at 3 seconds
}
