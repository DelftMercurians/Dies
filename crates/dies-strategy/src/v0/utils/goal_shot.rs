use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

pub fn has_clear_shot(s: &RobotSituation) -> bool {
    let shot_target = find_optimal_shot_target(s);

    // TODO: Implement proper ray casting when available
    // For now, simple distance check
    s.distance_to_position(shot_target) < 4000.0
}

pub fn find_optimal_shot_target(s: &RobotSituation) -> Vector2 {
    let goal_pos = s.get_opp_goal_position();

    if let Some(field) = &s.world.field_geom {
        let half_goal_width = field.goal_width / 2.0;

        // Simple strategy: alternate between corners based on player hash
        let hash = (s.player_id.as_u32() as f64 * 0.6180339887498949) % 1.0;
        if hash > 0.5 {
            Vector2::new(goal_pos.x, half_goal_width)
        } else {
            Vector2::new(goal_pos.x, -half_goal_width)
        }
    } else {
        goal_pos
    }
}

pub fn get_heading_toward_ball(s: &RobotSituation) -> f64 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let my_pos = s.player_data().position;
        (ball_pos - my_pos).y.atan2((ball_pos - my_pos).x)
    } else {
        0.0
    }
}

pub fn get_heading_to_goal(s: &RobotSituation) -> f64 {
    let player_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();
    (goal_pos - player_pos).y.atan2((goal_pos - player_pos).x)
}
