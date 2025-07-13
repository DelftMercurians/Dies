use dies_core::{PlayerId, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::get_defender_heading;

pub fn build_harasser_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .dynamic(|s| {
            // Find all threatening opponents in our half
            let threats: Vec<_> = s
                .world
                .opp_players
                .iter()
                .filter(|p| {
                    p.position.x < 0.0 && (p.position - s.get_own_goal_position()).norm() < 4000.0
                })
                .collect();

            // Create behavior options for each threat
            let mut options = Vec::new();
            for (_i, opponent) in threats.iter().enumerate() {
                // Each option tries to mark this specific opponent
                let opponent_pos = opponent.position;
                let opponent_id = opponent.id;
                options.push(
                    semaphore_node()
                        .do_then(
                            go_to_position(Argument::callback(move |s| {
                                calculate_harasser_position_for_pos(s, opponent_pos)
                            }))
                            .with_heading(Argument::callback(get_defender_heading))
                            .description(format!("Mark opponent {}", opponent_id))
                            .build(),
                        )
                        .semaphore_id(format!("tag_opponent_{}", opponent_id))
                        .max_entry(1)
                        .build()
                        .into(),
                );
            }

            // Add fallback behavior if no threats to mark
            if options.is_empty() {
                options.push(
                    go_to_position(Argument::callback(fallback_defender_position))
                        .with_heading(Argument::callback(get_defender_heading))
                        .description("Defend position".to_string())
                        .build()
                        .into(),
                );
            }

            options
        })
        .into()
}

pub fn score_as_harasser(s: &RobotSituation) -> f64 {
    let mut score = 40.0;

    // Find unmarked opponents in our half
    let unmarked_threats = find_unmarked_threats(s);
    if unmarked_threats.is_empty() {
        return 0.0; // No need for harasser
    }

    // Score based on proximity to highest threat opponent
    if let Some(highest_threat) = find_highest_threat_opponent(s, &unmarked_threats) {
        let dist_to_threat = s.distance_to_position(highest_threat.position);
        score += (2000.0 - dist_to_threat.min(2000.0)) / 40.0;

        // Bonus if we're the closest defender
        if is_closest_defender_to(s, highest_threat.position) {
            score += 20.0;
        }
    }

    score
}

fn find_unmarked_threats(s: &RobotSituation) -> Vec<&dies_core::PlayerData> {
    let opponents_in_our_half: Vec<_> = s
        .world
        .opp_players
        .iter()
        .filter(|p| p.position.x < 0.0)
        .collect();

    let mut unmarked = Vec::new();

    for opponent in opponents_in_our_half {
        if !is_opponent_marked(s, opponent) {
            unmarked.push(opponent);
        }
    }

    unmarked
}

/// Find highest threat opponent from list
fn find_highest_threat_opponent<'a>(
    s: &RobotSituation,
    opponents: &[&'a dies_core::PlayerData],
) -> Option<&'a dies_core::PlayerData> {
    opponents
        .iter()
        .max_by_key(|&&opponent| (evaluate_opponent_threat(s, opponent) * 1000.0) as i64)
        .copied()
}

/// Evaluate threat level of an opponent
fn evaluate_opponent_threat(s: &RobotSituation, opponent: &dies_core::PlayerData) -> f64 {
    // Distance to our goal
    let goal_dist = (opponent.position - s.get_own_goal_position()).norm();
    let dist_threat = (3000.0 - goal_dist.min(3000.0)) / 3000.0;

    // Distance to ball
    let ball_dist = if let Some(ball) = &s.world.ball {
        (opponent.position - ball.position.xy()).norm()
    } else {
        f64::INFINITY
    };
    let ball_threat = (1500.0 - ball_dist.min(1500.0)) / 1500.0;

    // Central position is more threatening
    let central_threat = 1.0 - (opponent.position.y.abs() / 3000.0).min(1.0);

    dist_threat * 0.5 + ball_threat * 0.3 + central_threat * 0.2
}

/// Check if we're the closest defender to a position
fn is_closest_defender_to(s: &RobotSituation, target_pos: Vector2) -> bool {
    let my_dist = (s.player_data().position - target_pos).norm();

    s.world
        .own_players
        .iter()
        .filter(|p| p.id != s.player_id && p.id != PlayerId::new(0)) // Exclude self and goalkeeper
        .all(|p| (p.position - target_pos).norm() >= my_dist)
}

/// Check if opponent is already marked by a teammate
fn is_opponent_marked(s: &RobotSituation, opponent: &dies_core::PlayerData) -> bool {
    let marking_distance = 800.0;

    s.world
        .own_players
        .iter()
        .any(|p| (p.position - opponent.position).norm() < marking_distance)
}

fn calculate_harasser_position_for_pos(s: &RobotSituation, target_pos: Vector2) -> Vector2 {
    let my_pos = s.player_data().position;
    let goal_pos = s.get_opp_goal_position();

    // Position to harass from the side toward goal
    let to_goal = (goal_pos - target_pos).normalize();
    let harass_offset = 800.0; // Distance to maintain

    // Choose side based on current position
    let perpendicular = Vector2::new(-to_goal.y, to_goal.x);
    let left_pos = target_pos + perpendicular * harass_offset;
    let right_pos = target_pos - perpendicular * harass_offset;

    // Choose closer side
    if (left_pos - my_pos).norm() < (right_pos - my_pos).norm() {
        left_pos
    } else {
        right_pos
    }
}

fn fallback_defender_position(s: &RobotSituation) -> Vector2 {
    let goal_pos = s.get_own_goal_position();

    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let ball_to_goal = goal_pos - ball_pos;

        // Position between ball and goal
        let defend_ratio = 0.4; // 40% of the way from ball to goal
        ball_pos + ball_to_goal * defend_ratio
    } else {
        goal_pos + Vector2::new(1500.0, 0.0)
    }
}
