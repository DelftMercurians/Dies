use dies_core::{Angle, Vector2};
use dies_executor::{behavior_tree_api::*, best_goal_shoot};

pub fn has_clear_shot(s: &RobotSituation) -> bool {
    let (_, p) = best_goal_shoot(&s.into());
    p > 0.3
}

pub fn find_optimal_shot_target(s: &RobotSituation) -> Vector2 {
    let (target, _) = best_goal_shoot(&s.into());
    dies_core::debug_circle_fill(
        "shooting target",
        target,
        100.0,
        dies_core::DebugColor::Purple,
    );

    target
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
