use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn has_clear_shot(s: &RobotSituation) -> bool {
    let shot_target = find_optimal_shot_target(s);

    // TODO: Implement proper ray casting when available
    // For now, simple distance check
    s.distance_to_position(shot_target) < 4000.0
}

pub fn find_optimal_shot_target(s: &RobotSituation) -> Vector2 {
    let goal_pos = s.get_opp_goal_position();
    // TODO: Implement proper shot target selection
    goal_pos
}

pub fn get_heading_toward_ball(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let my_pos = s.player_data().position;
        Angle::between_points(ball_pos, my_pos)
    } else {
        Angle::from_radians(0.0)
    }
}

pub fn get_heading_to_goal(s: &RobotSituation) -> Angle {
    let player_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();
    Angle::between_points(goal_pos, player_pos)
}
