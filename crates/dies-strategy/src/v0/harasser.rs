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
                                .build(),
                        )
                        .semaphore_id("harasser_ball_pickup".to_string())
                        .max_entry(1)
                        .build(),
                )
                .build(),
        )
        .add(select_node().dynamic(|s| {
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

            // Generate fallback behaviors for multiple harassers
            let fallback_options = generate_fallback_behaviors(s);
            options.extend(fallback_options);

            options
        }))
        .build()
        .into()
}

/// Generate multiple fallback behaviors for free harassers
fn generate_fallback_behaviors(s: &RobotSituation) -> Vec<BehaviorNode> {
    let mut behaviors = Vec::new();

    // Generate defensive positions for multiple harassers
    let defensive_positions = generate_defensive_positions(s);
    for (i, position) in defensive_positions.iter().enumerate() {
        let position_copy = *position;
        behaviors.push(
            semaphore_node()
                .do_then(
                    go_to_position(Argument::callback(move |_s| position_copy))
                        .with_heading(Argument::callback(get_defender_heading))
                        .description(format!("Defend position {}", i + 1))
                        .build(),
                )
                .semaphore_id(format!("harasser_defensive_pos_{}", i))
                .max_entry(1)
                .build()
                .into(),
        );
    }

    behaviors
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
            .any(|opp| (opp.position - ball_pos).norm() < 1000.0);

        if ball_threatened {
            return false;
        }

        // Check if any teammate is already close to the ball
        let teammate_nearby = s.world.own_players.iter().any(|teammate| {
            teammate.id != s.player_id && (teammate.position - ball_pos).norm() < 800.0
        });

        if teammate_nearby {
            return false;
        }

        // Check if ball is not moving fast (velocity magnitude < 500 mm/s)
        let ball_velocity = ball.velocity.xy();
        if ball_velocity.norm() > 500.0 {
            return false;
        }

        // Ball is free and on our side - we should pick it up
        true
    } else {
        false
    }
}

/// Generate multiple defensive positions for harassers
fn generate_defensive_positions(s: &RobotSituation) -> Vec<Vector2> {
    let mut positions = Vec::new();
    let goal_pos = s.get_own_goal_position();

    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();

        // Ensure ball position is constrained to our half for defensive calculations
        let defensive_ball_pos = Vector2::new(ball_pos.x.min(-200.0), ball_pos.y);

        // Primary defensive position - between ball and goal
        let primary_pos = calculate_primary_defensive_position(s, defensive_ball_pos, goal_pos);
        positions.push(primary_pos);

        // Secondary positions - flanking the primary position
        let secondary_positions =
            calculate_secondary_defensive_positions(s, defensive_ball_pos, goal_pos);
        positions.extend(secondary_positions);

        // Tertiary positions - wider defensive coverage
        let tertiary_positions = calculate_tertiary_defensive_positions(s, goal_pos);
        positions.extend(tertiary_positions);
    } else {
        // No ball - create default defensive positions
        positions.extend(calculate_default_defensive_positions(s, goal_pos));
    }

    // Ensure all positions are on our half and within field bounds
    positions
        .iter()
        .map(|&pos| constrain_to_our_half(s, pos))
        .collect()
}

/// Calculate primary defensive position between ball and goal
fn calculate_primary_defensive_position(
    s: &RobotSituation,
    ball_pos: Vector2,
    goal_pos: Vector2,
) -> Vector2 {
    let ball_to_goal = goal_pos - ball_pos;
    let defend_ratio = 0.4; // 40% of the way from ball to goal
    ball_pos + ball_to_goal * defend_ratio
}

/// Calculate secondary defensive positions flanking the primary
fn calculate_secondary_defensive_positions(
    s: &RobotSituation,
    ball_pos: Vector2,
    goal_pos: Vector2,
) -> Vec<Vector2> {
    let mut positions = Vec::new();
    let ball_to_goal = goal_pos - ball_pos;
    let ball_to_goal_normalized = ball_to_goal.normalize();

    // Create perpendicular vector for flanking
    let perpendicular = Vector2::new(-ball_to_goal_normalized.y, ball_to_goal_normalized.x);

    // Base position slightly closer to goal than primary
    let base_pos = ball_pos + ball_to_goal * 0.3;

    // Left and right flanking positions
    let flank_distance = 1200.0; // Distance from center line
    positions.push(base_pos + perpendicular * flank_distance);
    positions.push(base_pos - perpendicular * flank_distance);

    // Positions closer to goal for deeper coverage
    let deeper_pos = ball_pos + ball_to_goal * 0.6;
    positions.push(deeper_pos + perpendicular * 800.0);
    positions.push(deeper_pos - perpendicular * 800.0);

    positions
}

/// Calculate tertiary defensive positions for wider coverage
fn calculate_tertiary_defensive_positions(s: &RobotSituation, goal_pos: Vector2) -> Vec<Vector2> {
    let mut positions = Vec::new();

    // Wide defensive positions
    let wide_x = goal_pos.x + 2000.0; // 2m in front of goal
    let field_width = s
        .world
        .field_geom
        .as_ref()
        .map(|f| f.field_width)
        .unwrap_or(6000.0);
    let max_y = (field_width / 2.0 - 500.0).min(2000.0); // Stay within bounds

    positions.push(Vector2::new(wide_x, max_y)); // Top wide position
    positions.push(Vector2::new(wide_x, -max_y)); // Bottom wide position
    positions.push(Vector2::new(wide_x, 0.0)); // Center wide position

    // Mid-field defensive positions
    let mid_x = -1000.0; // 1m on our side
    positions.push(Vector2::new(mid_x, max_y / 2.0));
    positions.push(Vector2::new(mid_x, -max_y / 2.0));

    positions
}

/// Calculate default defensive positions when no ball is present
fn calculate_default_defensive_positions(s: &RobotSituation, goal_pos: Vector2) -> Vec<Vector2> {
    let mut positions = Vec::new();

    // Default formation spread in front of goal
    let base_x = goal_pos.x + 1500.0;
    let field_width = s
        .world
        .field_geom
        .as_ref()
        .map(|f| f.field_width)
        .unwrap_or(6000.0);
    let max_y = (field_width / 2.0 - 500.0).min(2000.0);

    positions.push(Vector2::new(base_x, 0.0)); // Center
    positions.push(Vector2::new(base_x, max_y)); // Top
    positions.push(Vector2::new(base_x, -max_y)); // Bottom
    positions.push(Vector2::new(base_x, max_y / 2.0)); // Mid-top
    positions.push(Vector2::new(base_x, -max_y / 2.0)); // Mid-bottom

    positions
}

/// Constrain position to our half of the field
fn constrain_to_our_half(s: &RobotSituation, pos: Vector2) -> Vector2 {
    let mut constrained_pos = pos;

    // Ensure we stay on our half (x < 0, but leave some margin)
    constrained_pos.x = constrained_pos.x.min(-200.0);

    // Constrain to field bounds
    if let Some(field) = &s.world.field_geom {
        let half_width = field.field_width / 2.0;
        let half_length = field.field_length / 2.0;

        constrained_pos.x = constrained_pos.x.max(-half_length + 100.0).min(-200.0);
        constrained_pos.y = constrained_pos
            .y
            .max(-half_width + 100.0)
            .min(half_width - 100.0);
    } else {
        // Default field bounds
        constrained_pos.x = constrained_pos.x.max(-4400.0).min(-200.0);
        constrained_pos.y = constrained_pos.y.max(-2900.0).min(2900.0);
    }

    constrained_pos
}

pub fn score_as_harasser(s: &RobotSituation) -> f64 {
    let mut score = 40.0;

    // Find unmarked opponents in our half
    let unmarked_threats = find_unmarked_threats(s);
    if unmarked_threats.is_empty() {
        // No threats to mark, but we can still be useful for fallback duties
        score = 20.0;

        // Bonus for being close to a good defensive position
        let defensive_positions = generate_defensive_positions(s);
        if let Some(closest_pos) = defensive_positions.first() {
            let dist_to_position = s.distance_to_position(*closest_pos);
            score += (1000.0 - dist_to_position.min(1000.0)) / 50.0;
        }

        // Bonus if we should pickup the ball
        if should_pickup_ball(s) {
            score += 15.0;
            let ball_pos = s
                .world
                .ball
                .as_ref()
                .map(|b| b.position.xy())
                .unwrap_or_default();
            let dist_to_ball = s.distance_to_position(ball_pos);
            score += (1500.0 - dist_to_ball.min(1500.0)) / 50.0;
        }

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
