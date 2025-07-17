use dies_core::{GameState, PlayerId, Vector2, PLAYER_RADIUS};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::fetch_and_shoot;

const SAFETY_MARGIN: f64 = 50.0; // Additional safety margin
const MIN_DISTANCE_TO_PLAYER: f64 = PLAYER_RADIUS * 2.0 + SAFETY_MARGIN;
const ARC_SAMPLES: usize = 12; // Number of positions to sample on arc

pub fn build_harasser_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            committing_guard_node()
                .when(|s| should_pickup_ball(s))
                .until(should_cancel_pickup_ball)
                .commit_to(
                    semaphore_node()
                        .semaphore_id("defender_pickup_ball".to_string())
                        .max_entry(1)
                        .do_then(fetch_and_shoot())
                        .build(),
                )
                .build(),
        )
        .add(
            scoring_select_node()
                .description("harasser positioning")
                .hysteresis_margin(0.1)
                .add_child(
                    semaphore_node()
                        .semaphore_id("primary_harasser".to_string())
                        .max_entry(1)
                        .do_then(
                            continuous("primary harasser")
                                .position(Argument::callback(move |s| {
                                    calculate_primary_harasser_position(s)
                                }))
                                .build(),
                        )
                        .build(),
                    |s| score_as_primary_harasser(s),
                )
                .add_child(
                    semaphore_node()
                        .semaphore_id("secondary_harasser".to_string())
                        .max_entry(1)
                        .do_then(
                            continuous("secondary harasser")
                                .position(Argument::callback(move |s| {
                                    calculate_secondary_harasser_position(s)
                                }))
                                .build(),
                        )
                        .build(),
                    |s| score_as_secondary_harasser(s),
                )
                .build(),
        )
        .build()
        .into()
}

/// Check if a harasser should go pickup the ball
fn should_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return false;
    }
    if !s.can_touch_ball() {
        return false;
    }

    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();

        // Only consider if ball is on our half
        if ball_pos.x >= 70.0 {
            return false;
        }

        // Check if we are closer to the ball than the closest opponent by a margin of 100mm
        let our_dist = s.distance_to_position(ball_pos);
        let closest_opp_dist = s.distance_of_closest_opp_player_to_ball();
        if our_dist > closest_opp_dist - 100.0 {
            return false;
        }

        // Check if ball is not moving fast (velocity magnitude < 500 mm/s)
        let ball_velocity = ball.velocity.xy();
        if ball_velocity.norm() > 500.0 {
            return false;
        }

        true
    } else {
        false
    }
}

fn should_cancel_pickup_ball(s: &RobotSituation) -> bool {
    s.ball_position().x > 100.0 || s.distance_of_closest_opp_player_to_ball() < 150.0
}

pub fn score_as_harasser(s: &RobotSituation) -> f64 {
    let mut score = 40.0;

    // Strongly prefer non-wallers
    if s.current_role_is("waller") {
        return 0.0;
    }

    // Find unmarked opponents in our half
    let unmarked_threats = find_unmarked_threats(s);
    if unmarked_threats.is_empty() {
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

fn calculate_primary_harasser_position(s: &RobotSituation) -> Vector2 {
    let ball_pos = s.ball_position();
    let own_goal = s.get_own_goal_position();

    // Check if closest opponent is within 0.5m of ball
    if let Some(closest_opp) = s.get_closest_opp_player_to_ball() {
        let opp_to_ball_dist = (closest_opp.position - ball_pos).norm();
        if opp_to_ball_dist < 500.0 {
            // Position on line from ball towards opponent heading
            let opp_heading = closest_opp.yaw.rotate_vector(&Vector2::x());
            let harass_distance = 300.0;
            let target_pos = closest_opp.position + opp_heading * harass_distance;
            let constrained_pos = s.constrain_to_field(target_pos);
            let final_pos = Vector2::new(constrained_pos.x.min(-50.0), constrained_pos.y);

            // Check if the position is obstructed
            if is_position_obstructed(s, final_pos) {
                // Sample positions on arc around opponent
                let angle_to_opp = opp_heading.y.atan2(opp_heading.x);
                return find_best_position_on_arc(
                    s,
                    closest_opp.position,
                    harass_distance,
                    angle_to_opp,
                    final_pos,
                );
            }

            let final_pos = Vector2::new(constrained_pos.x.min(-50.0), constrained_pos.y);

            return final_pos;
        }
    }

    // Default: position on line from ball towards our goal
    let harass_distance = 300.0;
    let to_goal = (own_goal - ball_pos).normalize();
    let target_pos = ball_pos + to_goal * harass_distance;

    // Check if the position is obstructed
    if is_position_obstructed(s, target_pos) {
        // Sample positions on arc around ball
        let angle_to_goal = to_goal.y.atan2(to_goal.x);
        let final_pos =
            find_best_position_on_arc(s, ball_pos, harass_distance, angle_to_goal, target_pos);
        return Vector2::new(final_pos.x.min(-50.0), final_pos.y);
    }

    // let constrained_pos = s.constrain_to_field(target_pos);
    let final_pos = Vector2::new(target_pos.x.min(-50.0), target_pos.y);

    final_pos
}

fn calculate_secondary_harasser_position(s: &RobotSituation) -> Vector2 {
    let ball_pos = s.ball_position();
    let harass_distance = 300.0;

    // Find unmarked opponent on our side
    let unmarked_threats = find_unmarked_threats(s);
    if !unmarked_threats.is_empty() {
        if let Some(target_opp) = find_highest_threat_opponent(s, &unmarked_threats) {
            // Position between opponent and ball, maintaining 30cm distance
            let to_ball = (ball_pos - target_opp.position).normalize();
            let target_pos = target_opp.position + to_ball * harass_distance;
            let constrained_pos = s.constrain_to_field(target_pos);
            // Ensure we stay on our side of the field (x < 0)
            return Vector2::new(constrained_pos.x.min(-50.0), constrained_pos.y);
        }
    }

    // No unmarked opponent on our side - position near second closest opponent to ball
    let mut opp_distances: Vec<_> = s
        .world
        .opp_players
        .iter()
        .map(|opp| (opp, (opp.position - ball_pos).norm()))
        .collect();
    opp_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    if opp_distances.len() >= 2 {
        let second_closest = opp_distances[1].0;
        // Find closest point on our side of field to this opponent
        let target_pos = if second_closest.position.x > 0.0 {
            // Opponent is on their side, position at center line but stay on our side
            Vector2::new(-50.0, second_closest.position.y)
        } else {
            // Opponent is on our side, position near them
            second_closest.position
        };
        let constrained_pos = s.constrain_to_field(target_pos);
        // Ensure we stay on our side of the field (x < 0)
        return Vector2::new(constrained_pos.x.min(-50.0), constrained_pos.y);
    }

    // Fallback: defensive position
    let own_goal = s.get_own_goal_position();
    let defense_line_x = 0.6 * own_goal.x;
    let constrained_pos = s.constrain_to_field(Vector2::new(defense_line_x, ball_pos.y));
    // Ensure we stay on our side of the field (x < 0)
    Vector2::new(constrained_pos.x.min(-50.0), constrained_pos.y)
}

fn score_as_primary_harasser(s: &RobotSituation) -> f64 {
    let mut score = 60.0; // Higher base score for primary role

    // Bonus for being close to ball
    let ball_dist = s.distance_to_ball();
    score += (1500.0 - ball_dist.min(1500.0)) / 30.0;

    // Bonus if there's a close opponent to harass
    if let Some(closest_opp) = s.get_closest_opp_player_to_ball() {
        let opp_to_ball_dist = (closest_opp.position - s.ball_position()).norm();
        if opp_to_ball_dist < 500.0 {
            score += 20.0;
        }
    }

    score
}

fn score_as_secondary_harasser(s: &RobotSituation) -> f64 {
    let mut score = 40.0; // Lower base score for secondary role

    // Check for unmarked threats
    let unmarked_threats = find_unmarked_threats(s);
    if !unmarked_threats.is_empty() {
        score += 25.0; // Bonus for having threats to mark

        // Additional bonus based on threat level
        if let Some(highest_threat) = find_highest_threat_opponent(s, &unmarked_threats) {
            let threat_score = evaluate_opponent_threat(s, highest_threat);
            score += threat_score * 20.0;
        }
    }

    score
}

/// Check if a position is obstructed by defense area, field boundaries, or other players
fn is_position_obstructed(s: &RobotSituation, pos: Vector2) -> bool {
    // Check if position is in own defense area (penalty area)
    if let Some(field) = &s.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        // Check own penalty area
        if pos.x <= -half_length + field.penalty_area_depth
            && pos.y >= -half_penalty_width
            && pos.y <= half_penalty_width
        {
            return true;
        }

        // Check opponent penalty area
        if pos.x >= half_length - field.penalty_area_depth
            && pos.y >= -half_penalty_width
            && pos.y <= half_penalty_width
        {
            return true;
        }
    }

    // Check if position is too close to field boundaries
    let constrained_pos = s.constrain_to_field(pos);
    if (constrained_pos - pos).norm() > 10.0 {
        return true;
    }

    // Check if position is too close to other players
    for player in &s.world.own_players {
        if player.id != s.player_id {
            if (player.position - pos).norm() < MIN_DISTANCE_TO_PLAYER {
                return true;
            }
        }
    }

    for player in &s.world.opp_players {
        if (player.position - pos).norm() < MIN_DISTANCE_TO_PLAYER {
            return true;
        }
    }

    false
}

/// Score a position based on distance to original target (primary) and centrality (secondary)
fn score_position(pos: Vector2, target: Vector2) -> f64 {
    let distance_to_target = (pos - target).norm();
    let field_center = Vector2::new(0.0, 0.0);
    let centrality_score = (pos - field_center).norm();

    // Primary preference: closest to original target
    // Secondary preference: closer to field center
    distance_to_target + centrality_score * 0.1
}

/// Sample positions on an arc around a center point and return the best valid position
fn find_best_position_on_arc(
    s: &RobotSituation,
    center: Vector2,
    radius: f64,
    preferred_angle: f64,
    original_target: Vector2,
) -> Vector2 {
    use std::f64::consts::PI;

    let mut best_pos = original_target;
    let mut best_score = f64::INFINITY;
    let mut found_valid = false;

    // Sample positions around the arc
    for i in 0..ARC_SAMPLES {
        let angle_offset = (i as f64 / ARC_SAMPLES as f64) * 2.0 * PI;
        let angle = preferred_angle + angle_offset;
        let pos = center + Vector2::new(angle.cos(), angle.sin()) * radius;

        if !is_position_obstructed(s, pos) {
            found_valid = true;
            let score = score_position(pos, original_target);

            if score < best_score {
                best_score = score;
                best_pos = pos;
            }
        }
    }

    // If no valid position found, try larger radius
    if !found_valid && radius < 1000.0 {
        return find_best_position_on_arc(
            s,
            center,
            radius * 1.3,
            preferred_angle,
            original_target,
        );
    }

    // Ensure the position is within field boundaries
    s.constrain_to_field(best_pos)
}
