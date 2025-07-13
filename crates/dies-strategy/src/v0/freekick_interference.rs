use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::get_defender_heading;

pub fn build_free_kick_interference_tree(_s: &RobotSituation) -> BehaviorNode {
    go_to_position(calculate_free_kick_defense_position)
        .with_heading(get_defender_heading)
        .description("Free kick defense")
        .build()
        .into()
}

pub fn score_free_kick_interference(s: &RobotSituation) -> f64 {
    let mut score = 70.0;

    // Prefer robots that can position between ball and our goal
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();
        let my_pos = s.player_data().position;

        // Calculate positioning score
        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_me = my_pos - ball_pos;
        let projection =
            (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
        let projection_ratio = projection / ball_to_goal.norm();

        if projection_ratio > 0.2 && projection_ratio < 0.8 {
            score += 20.0;
        }
    }

    score
}

fn calculate_free_kick_defense_position(s: &RobotSituation) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();
        let ball_to_goal = (goal_pos - ball_pos).normalize();

        // Position 700mm from ball toward goal (SSL rule: 500mm minimum)
        ball_pos + ball_to_goal * 700.0
    } else {
        s.get_own_goal_position() + Vector2::new(1000.0, 0.0)
    }
}
