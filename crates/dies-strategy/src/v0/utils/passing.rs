use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

pub fn find_best_pass_target(s: &RobotSituation) -> Vector2 {
    let teammates = &s.world.own_players;
    let my_id = s.player_id;

    let mut best_target = s.get_opp_goal_position();
    let mut best_score = -1.0;

    for teammate in teammates {
        if teammate.id == my_id {
            continue;
        }

        // Score based on goal distance
        let goal_dist = s.distance_to_position(teammate.position);
        let score = 10000.0 - goal_dist;

        if score > best_score {
            best_score = score;
            best_target = teammate.position;
        }
    }

    best_target
}

pub fn can_pass_to_teammate(s: &RobotSituation) -> bool {
    if !s.has_ball() {
        return false;
    }

    let teammates = &s.world.own_players;
    teammates.iter().any(|p| p.id != s.player_id)
}
