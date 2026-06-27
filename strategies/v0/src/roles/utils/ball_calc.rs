use dies_core::Vector2;

use crate::bt::RobotSituation;

pub fn calculate_intercept_position(s: &RobotSituation) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        if ball.velocity.norm() < 100.0 {
            return ball.position;
        }

        let ball_pos = ball.position;
        let ball_vel = ball.velocity;
        let player_pos = s.position();

        let intercept_time = estimate_intercept_time(ball_pos, ball_vel, player_pos);
        ball_pos + ball_vel * intercept_time
    } else {
        s.position()
    }
}

fn estimate_intercept_time(ball_pos: Vector2, ball_vel: Vector2, player_pos: Vector2) -> f64 {
    let max_robot_speed = 3000.0; // mm/s

    let mut time_estimate = 0.5;
    for _ in 0..5 {
        let predicted_ball = ball_pos + ball_vel * time_estimate;
        let distance = (predicted_ball - player_pos).norm();
        time_estimate = distance / max_robot_speed;
    }

    time_estimate.min(3.0)
}
