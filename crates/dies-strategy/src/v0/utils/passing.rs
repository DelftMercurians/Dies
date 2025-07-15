use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn clean_goal_shot_score(s: &RobotSituation, player: &dies_core::PlayerData) -> f64 {
    let player_pos = player.position;
    let goal_pos = s.get_opp_goal_position();

    // Get goal geometry
    let geom = s.world.field_geom.clone().unwrap();
    let goal_width = geom.goal_width;

    // Calculate goal boundaries
    let goal_left = Vector2::new(goal_pos.x, goal_pos.y - goal_width / 2.0);
    let goal_right = Vector2::new(goal_pos.x, goal_pos.y + goal_width / 2.0);

    // Factor 1: Distance to goal (closer is better, normalized between 0-1)
    let distance_to_goal = (player_pos - goal_pos).norm();
    let max_field_distance = 8000.0; // Approximate max field distance
    let distance_score = 1.0 - (distance_to_goal / max_field_distance).min(1.0);

    // Factor 2: Visibility angle of the goal (wider angle is better)
    let left_angle = Angle::between_points(goal_left, player_pos);
    let right_angle = Angle::between_points(goal_right, player_pos);
    let visibility_angle = (left_angle.radians() - right_angle.radians()).abs();
    let max_visibility_angle = std::f64::consts::PI;
    let visibility_score = (visibility_angle / max_visibility_angle).min(1.0);

    // Factor 3: Check if shot line is blocked
    let center_angle = Angle::between_points(player_pos, goal_pos);
    let mut temp_situation = s.clone();
    temp_situation.player_id = player.id;
    let nearest_opponent_distance =
        find_nearest_opponent_distance_along_direction(&temp_situation, center_angle);

    let blocking_score = if nearest_opponent_distance < 200.0 {
        0.0
    } else if nearest_opponent_distance < 500.0 {
        0.3
    } else if nearest_opponent_distance < 1000.0 {
        0.7
    } else {
        1.0
    };

    // println!("{:.2} {:.2} {:.2}", distance_score, visibility_score, blocking_score);
    // Weighted combination of factors
    let score = distance_score * 0.2 + visibility_score * 0.3 + blocking_score * 0.5;
    score.clamp(0.0, 1.0)
}

pub fn find_best_shoot_target(s: &RobotSituation) -> Vector2 {
    let teammates = &s.world.own_players;
    let my_id = s.player_id;

    let mut best_target = s.get_opp_goal_position();
    let mut best_score = clean_goal_shot_score(s, s.player_data());
    // println!("{} scored as {}", s.player_id, best_score);

    for teammate in teammates {
        if teammate.id == my_id {
            continue;
        }
        let robot_pos = s.player_data().position;

        // Score based on goal distance
        let clean_shot_score = clean_goal_shot_score(s, teammate);
        let mut score = clean_shot_score;

        // score based on how far is the robot: not too close, not too far
        let robot_dist = (robot_pos - teammate.position).norm();
        let dist_badness = if robot_dist < 400.0 {
            0.2
        } else if robot_dist < 800.0 {
            0.1
        } else if robot_dist < 1200.0 {
            0.05
        } else if robot_dist < 3000.0 {
            0.0 // this is totally fine
        } else if robot_dist < 5000.0 {
            0.3 // this is mid
        } else {
            0.5 // meh
        };
        score -= dist_badness;

        // score based on how bad is the trajectory: are there opponents on the shot line?
        let angle = Angle::between_points(teammate.position, robot_pos);
        let nearest_opponent_distance =
            find_nearest_opponent_distance_along_direction(s, angle).clamp(0.0, 1000.0);
        let low = 250.0;
        let high = 500.0;
        if nearest_opponent_distance < low {
            score -= 0.5;
        } else if nearest_opponent_distance < high {
            score -= 0.1;
        } else {
            score -= 0.0;
        }

        // println!("{} passing {}: {:.2}; clean: {:.2}", s.player_id, teammate.id, score, clean_shot_score);

        // Stringly prefer passing to strikers
        let teammate_role = s
            .role_assignments
            .get(&teammate.id)
            .cloned()
            .unwrap_or_default();
        let teammate_striker = teammate_role == "striker";
        if teammate_striker {
            score += 0.2;
        }

        if score > best_score {
            best_score = score;
            best_target = teammate.position;
            // println!("passing to an opponent! {} {}", best_target, best_score);
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

pub fn find_nearest_opponent_distance_along_direction(s: &RobotSituation, direction: Angle) -> f64 {
    let robot_pos = s.player_data().position;
    let direction_vector = direction.to_vector();

    let mut min_distance = f64::INFINITY;

    // Check all opponent robots
    for player in s.world.opp_players.iter() {
        if player.id != s.player_data().id {
            let opp_pos = player.position;
            let to_opponent = opp_pos - robot_pos;

            // Project opponent position onto the shooting direction
            let projection = to_opponent.dot(&direction_vector);

            // Only consider opponents in front of us
            if projection > 0.0 {
                let perpendicular_distance = (to_opponent - projection * direction_vector).norm();

                // Consider robot radius (approximate as 90mm)
                let effective_distance = perpendicular_distance - 90.0;
                min_distance = min_distance.min(effective_distance.max(0.0));
            }
        }
    }

    min_distance
}
