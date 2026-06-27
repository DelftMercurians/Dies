use dies_core::{Angle, Vector2};

use crate::bt::*;

pub fn build_free_kick_interference_tree(_s: &RobotSituation) -> BehaviorNode {
    go_to_position(calculate_free_kick_defense_position as fn(&RobotSituation) -> Vector2)
        .with_heading(head_towards_ball as fn(&RobotSituation) -> Angle)
        .description("Free kick defense")
        .build()
        .into()
}

fn calculate_free_kick_defense_position(s: &RobotSituation) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position;
        let goal_pos = s.get_own_goal_position();
        let ball_to_goal = (goal_pos - ball_pos).normalize();

        // 700mm from the ball toward our goal (SSL rule: 500mm minimum).
        ball_pos + ball_to_goal * 700.0
    } else {
        s.get_own_goal_position() + Vector2::new(1200.0, 0.0)
    }
}

pub fn head_towards_ball(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        Angle::between_points(s.position(), ball.position)
    } else {
        Angle::from_radians(0.0)
    }
}
