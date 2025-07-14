use dies_core::{PlayerId, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{find_best_pass_target, get_defender_heading};

pub fn build_harasser_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            guard_node()
                .condition(|s| should_pickup_ball(s))
                .then(
                    semaphore_node()
                        .do_then(
                            select_node()
                                .add(
                                    fetch_ball()
                                        .description("Pickup free ball".to_string())
                                        .build(),
                                )
                                .add(face_position(find_best_pass_target).build())
                                .add(kick().build())
                                .build(),
                        )
                        .semaphore_id("harasser_ball_pickup".to_string())
                        .max_entry(1)
                        .build(),
                )
                .build(),
        )
        .add(
            continuous("standby position 1")
                .position(Argument::callback(move |s| {
                    // Defense line: 60% of the way from our goal to halfway line (on our side)
                    let own_goal = s.get_own_goal_position();
                    let defense_line_x = 0.6 * own_goal.x;
                    // If ball is moving, project intersection with defense line
                    if let Some(ball) = &s.world.ball {
                        let ball_pos = ball.position.xy();
                        let ball_vel = ball.velocity.xy();
                        // Only if ball is moving toward our goal (negative x)
                        if ball_vel.x < -10.0 {
                            // t = (defense_line_x - ball_pos.x) / ball_vel.x
                            let t = (defense_line_x - ball_pos.x) / ball_vel.x;
                            if t > 0.0 && t < 5.0 {
                                // Projected intersection point
                                let intersection = ball_pos + ball_vel * t;
                                return intersection;
                            }
                        } else {
                            return Vector2::new(defense_line_x, ball_pos.y);
                        }
                    }
                    let y = 0.0;
                    Vector2::new(defense_line_x, y)
                }))
                .build(),
        )
        // .add(select_node().dynamic(|s| {
        //     // Find all threatening opponents in our half
        //     let threats: Vec<_> = s
        //         .world
        //         .opp_players
        //         .iter()
        //         .filter(|p| {
        //             p.position.x < 0.0 && (p.position - s.get_own_goal_position()).norm() < 4000.0
        //         })
        //         .collect();
        //     // Create behavior options for each threat
        //     let mut options: Vec<BehaviorNode> = Vec::new();
        //     for (_i, opponent) in threats.iter().enumerate() {
        //         // Each option tries to mark this specific opponent
        //         let opponent_pos = opponent.position;
        //         let opponent_id = opponent.id;
        //         options.push(
        //             semaphore_node()
        //                 .do_then(
        //                     continuous(format!("mark opponent {}", opponent_id))
        //                         .position(Argument::callback(move |s| {
        //                             calculate_harasser_position_for_pos(s, opponent_pos)
        //                         }))
        //                         .heading(Argument::callback(get_defender_heading))
        //                         .build(),
        //                 )
        //                 .semaphore_id(format!("tag_opponent_{}", opponent_id))
        //                 .max_entry(1)
        //                 .build()
        //                 .into(),
        //         );
        //     }
        //     // 1. Standby position
        //     options.push(
        //         continuous("standby position 1")
        //             .position(Argument::callback(move |s| {
        //                 // Defense line: 60% of the way from our goal to halfway line (on our side)
        //                 let own_goal = s.get_own_goal_position();
        //                 let defense_line_x = 0.6 * own_goal.x;
        //                 // If ball is moving, project intersection with defense line
        //                 if let Some(ball) = &s.world.ball {
        //                     let ball_pos = ball.position.xy();
        //                     let ball_vel = ball.velocity.xy();
        //                     // Only if ball is moving toward our goal (negative x)
        //                     if ball_vel.x < -10.0 {
        //                         // t = (defense_line_x - ball_pos.x) / ball_vel.x
        //                         let t = (defense_line_x - ball_pos.x) / ball_vel.x;
        //                         if t > 0.0 && t < 5.0 {
        //                             // Projected intersection point
        //                             let intersection = ball_pos + ball_vel * t;
        //                             return intersection;
        //                         }
        //                     } else {
        //                         return Vector2::new(defense_line_x, ball_pos.y);
        //                     }
        //                 }
        //                 let y = 0.0;
        //                 Vector2::new(defense_line_x, y)
        //             }))
        //             .build()
        //             .into(),
        //     );
        //     // 2. Standby position
        //     options.push(
        //         continuous("standby position 2")
        //             .position(Argument::callback(move |s| {
        //                 // Defense line: 60% of the way from our goal to halfway line (on our side)
        //                 let own_goal = s.get_own_goal_position();
        //                 let halfway_x = 0.0;
        //                 let defense_line_x = own_goal.x + 0.6 * (halfway_x - own_goal.x);
        //                 // Shadow closest opponent on the defense line
        //                 let closest_opponent = s.find_opp_player_min_by(|p| p.position.x);
        //                 if let Some(opponent) = closest_opponent {
        //                     return opponent.position;
        //                 }
        //                 Vector2::new(defense_line_x, 0.0)
        //             }))
        //             .build()
        //             .into(),
        //     );
        //     options
        // }))
        .build()
        .into()
}

/// Check if a harasser should go pickup the ball
fn should_pickup_ball(s: &RobotSituation) -> bool {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();

        // Only consider if ball is on our half
        if ball_pos.x >= 0.0 {
            return false;
        }

        // Check if ball is close to any opponent (within 1000mm)
        let ball_threatened = s
            .world
            .opp_players
            .iter()
            .any(|opp| (opp.position - ball_pos).norm() < 500.0);

        if ball_threatened {
            println!("{} ball threatened", s.viz_path_prefix);
            return false;
        }

        // Check if any teammate is already close to the ball
        let teammate_nearby = s.world.own_players.iter().any(|teammate| {
            teammate.id != s.player_id && (teammate.position - ball_pos).norm() < 500.0
        });

        if teammate_nearby {
            println!("{} teammate nearby", s.viz_path_prefix);
            return false;
        }

        // Check if ball is not moving fast (velocity magnitude < 500 mm/s)
        let ball_velocity = ball.velocity.xy();
        if ball_velocity.norm() > 500.0 {
            println!("{} ball velocity too high", s.viz_path_prefix);
            return false;
        }

        // Ball is free and on our side - we should pick it up
        true
    } else {
        false
    }
}

pub fn score_as_harasser(s: &RobotSituation) -> f64 {
    let mut score = 40.0;

    // Find unmarked opponents in our half
    let unmarked_threats = find_unmarked_threats(s);
    if unmarked_threats.is_empty() {
        // No threats to mark, but we can still be useful for fallback duties
        score = 10.0;

        // Bonus for being close to a good defensive position
        // let defensive_positions = generate_defensive_positions(s);
        // if let Some(closest_pos) = defensive_positions.first() {
        //     let dist_to_position = s.distance_to_position(*closest_pos);
        //     score += (1000.0 - dist_to_position.min(1000.0)) / 50.0;
        // }

        // // Bonus if we should pickup the ball
        // if should_pickup_ball(s) {
        //     score += 15.0;
        //     let ball_pos = s
        //         .world
        //         .ball
        //         .as_ref()
        //         .map(|b| b.position.xy())
        //         .unwrap_or_default();
        //     let dist_to_ball = s.distance_to_position(ball_pos);
        //     score += (1500.0 - dist_to_ball.min(1500.0)) / 50.0;
        // }

        return score;
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
