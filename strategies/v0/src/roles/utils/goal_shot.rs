use dies_core::{Angle, Vector2};

use crate::bt::RobotSituation;
use crate::helpers::best_goal_shoot;

pub fn has_clear_shot(s: &RobotSituation) -> bool {
    best_goal_shoot(s).1 > 0.3
}

pub fn find_optimal_shot_target(s: &RobotSituation) -> Vector2 {
    let (target, _) = best_goal_shoot(s);
    target
}

pub fn get_heading_toward_ball(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        Angle::between_points(ball.position, s.position())
    } else {
        Angle::from_radians(0.0)
    }
}

pub fn get_heading_to_goal(s: &RobotSituation) -> Angle {
    Angle::between_points(s.get_opp_goal_position(), s.position())
}
