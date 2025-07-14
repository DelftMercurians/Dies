use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;
use super::super::utils::find_nearest_opponent_distance_along_direction;

pub fn has_clear_shot(s: &RobotSituation) -> bool {
    let (score, _) = choose_best_direction_and_score(s);
    score > 0.0
}

pub fn find_optimal_shot_target(s: &RobotSituation) -> Vector2 {
    let (_, best_direction) = choose_best_direction_and_score(s);
    let robot_pos = s.player_data().position;

    // Calculate intersection with goal line
    let goal_pos = s.get_opp_goal_position();
    let direction_vector = best_direction.to_vector();

    // Find intersection with goal line (x = goal_pos.x)
    let t = (goal_pos.x - robot_pos.x) / direction_vector.x;
    let intersection_y = robot_pos.y + t * direction_vector.y;

    let target = Vector2::new(goal_pos.x, intersection_y);

    dies_core::debug_circle_fill("shooting target", target, 100.0, dies_core::DebugColor::Purple);

    target
}

pub fn choose_best_direction_and_score(s: &RobotSituation) -> (f64, Angle) {
    let robot_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();

    // Get goal geometry
    let geom = s.world.field_geom.clone().unwrap();
    let goal_width = geom.goal_width;

    // Calculate goal boundaries
    let goal_left = Vector2::new(goal_pos.x, goal_pos.y - goal_width / 2.0);
    let goal_right = Vector2::new(goal_pos.x, goal_pos.y + goal_width / 2.0);

    // Calculate angle range for sampling
    let left_angle = Angle::between_points(goal_left, robot_pos);
    let right_angle = Angle::between_points(goal_right, robot_pos);

    // Sample angles between goal boundaries
    let mut best_score = 0.0;
    let mut best_angle = Angle::between_points(goal_pos, robot_pos);

    let num_samples = 50;
    println!("starting");
    for i in 0..num_samples {
        let t = i as f64 / (num_samples - 1) as f64;
        let angle = left_angle.lerp(right_angle, t);
        println!("{}", angle);

        let score = score_shot_direction(s, angle);
        if score > best_score {
            best_score = score;
            best_angle = angle;
        }
    }

    (best_score, best_angle)
}

fn score_shot_direction(s: &RobotSituation, direction: Angle) -> f64 {
    let robot_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();

    // Find nearest opponent robot distance along this direction
    let nearest_opponent_distance = find_nearest_opponent_distance_along_direction(s, Angle::PI - direction);

    // Score is zero if distance < 200mm, then increases quadratically
    if nearest_opponent_distance < 200.0 {
        return 0.0;
    }

    let mut score = 1.0;

    // Factor 1: Distance to nearest opponent (larger is better, quadratic growth)
    let distance_factor = ((nearest_opponent_distance - 200.0) / 1000.0).powi(2);
    score += distance_factor;

    // Factor 2: Angle to middle of goal (closer to center is better)
    let center_angle = Angle::between_points(goal_pos, robot_pos);
    let angle_diff = (direction.radians() - center_angle.radians()).abs();
    let angle_factor = (1.0 - (angle_diff / std::f64::consts::PI)).min(1.0);
    score -= angle_factor * 5.0;

    // Factor 3: Distance preference (closer intersection with goal line)
    let direction_vector = direction.to_vector();
    let t = (goal_pos.x - robot_pos.x) / direction_vector.x;
    let intersection_y = robot_pos.y + t * direction_vector.y;
    let intersection = Vector2::new(goal_pos.x, intersection_y);
    let distance_to_intersection = (robot_pos - intersection).norm();
    score -= distance_to_intersection / 3000.0;

    score
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
