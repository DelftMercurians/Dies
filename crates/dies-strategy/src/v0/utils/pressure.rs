use dies_executor::behavior_tree_api::*;

pub fn under_pressure(s: &RobotSituation) -> bool {
    s.get_own_players_within_radius_of_me(800.0).len() > 0
}

pub fn evaluate_ball_threat(s: &RobotSituation) -> f64 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();

        // Distance factor
        let ball_goal_dist = (ball_pos - goal_pos).norm();
        let dist_threat = (4000.0 - ball_goal_dist.min(4000.0)) / 4000.0;

        // Central position factor
        let central_threat = 1.0 - (ball_pos.y.abs() / 3000.0).min(1.0);

        // Ball in our half factor
        let half_factor = if ball_pos.x < 0.0 { 1.0 } else { 0.3 };

        dist_threat * central_threat * half_factor
    } else {
        0.0
    }
}
