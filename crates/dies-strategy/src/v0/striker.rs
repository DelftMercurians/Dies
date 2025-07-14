use dies_core::{Angle, GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{
    can_pass_to_teammate, find_best_pass_target, find_optimal_shot_target, get_heading_toward_ball,
    has_clear_shot, under_pressure,
};

pub fn build_striker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Handle kickoff positioning - stay on our side
            guard_node()
                .condition(|s| {
                    s.game_state_is_one_of(&[GameState::PrepareKickoff, GameState::Kickoff])
                })
                .then(build_kickoff_positioning_behavior())
                .description("Kickoff positioning")
                .build(),
        )
        .add(
            guard_node()
                .condition(|s| should_pickup_ball(s) && s.am_closest_to_ball())
                .description("Ball pickup opportunity")
                .then(
                    semaphore_node()
                        .do_then(
                            sequence_node()
                                .add(
                                    fetch_ball()
                                        .description("Pickup free ball".to_string())
                                        .build(),
                                )
                                .add(face_position(find_best_pass_target).with_ball().build())
                                .add(kick().build())
                                .build(),
                        )
                        .semaphore_id("striker_ball_pickup".to_string())
                        .max_entry(1)
                        .build(),
                )
                .build(),
        )
        .add(
            // Normal striker behavior with zone allocation
            build_zone_based_striker_behavior(),
        )
        .description("Striker")
        .build()
        .into()
}

pub fn score_striker(s: &RobotSituation) -> f64 {
    let mut score = 50.0;

    // Prefer robots closer to ball
    let ball_dist = s.distance_to_ball();
    score += (2000.0 - ball_dist.min(2000.0)) / 20.0;

    // Prefer robots in attacking position
    if s.player_data().position.x > 0.0 {
        score += 20.0;
    }

    // Higher score if we have the ball
    if s.has_ball() {
        score += 30.0;
    }

    score
}

fn build_kickoff_positioning_behavior() -> BehaviorNode {
    go_to_position(get_kickoff_striker_position)
        .with_heading(get_heading_toward_ball)
        .description("Kickoff positioning")
        .build()
        .into()
}

fn build_zone_based_striker_behavior() -> BehaviorNode {
    scoring_select_node()
        .add_child(build_striker_in_zone("top"), |s| score_for_zone(s, "top"))
        .add_child(build_striker_in_zone("middle"), |s| {
            score_for_zone(s, "middle")
        })
        .add_child(build_striker_in_zone("bottom"), |s| {
            score_for_zone(s, "bottom")
        })
        .hysteresis_margin(0.2) // Increased hysteresis for zone selection
        .description("Choose Attack Zone")
        .build()
        .into()
}

/// Build striker behavior within a specific zone
fn build_striker_in_zone(zone: &str) -> BehaviorNode {
    let zone_owned = zone.to_string();
    scoring_select_node()
        .add_child(
            guard_node()
                .condition(|s| s.has_ball())
                .then(build_ball_carrier_behavior(&zone_owned))
                .description("Have ball")
                .build()
                .into(),
            |s| if s.has_ball() { 100.0 } else { 0.0 },
        )
        .add_child(
            fetch_ball()
                .description("Get ball".to_string())
                .build()
                .into(),
            |s| {
                let ball_dist = s.distance_to_ball();
                if ball_dist < 1000.0 {
                    80.0 - ball_dist / 20.0
                } else {
                    0.0
                }
            },
        )
        .add_child(
            {
                let zone_inner = zone_owned.clone();
                {
                    let zone_name = zone_inner.clone();
                    go_to_position(Argument::callback(move |s| {
                        find_optimal_striker_position(s, &zone_inner)
                    }))
                    .with_heading(Argument::callback(|s: &RobotSituation| {
                        if let Some(ball) = &s.world.ball {
                            let ball_pos = ball.position.xy();
                            let my_pos = s.player_data().position;
                            Angle::between_points(ball_pos, my_pos)
                        } else {
                            Angle::from_radians(0.0)
                        }
                    }))
                    .description(format!("Position in {}", zone_name))
                    .build()
                    .into()
                }
            },
            |_| 30.0,
        )
        .hysteresis_margin(0.2)
        .description(format!("Zone {} Actions", zone))
        .build()
        .into()
}

/// Build ball carrier decision making behavior
fn build_ball_carrier_behavior(zone: &str) -> BehaviorNode {
    scoring_select_node()
        .add_child(build_shoot_sequence(), score_for_shooting)
        .add_child(build_pass_sequence(), score_for_passing)
        .add_child(
            build_dribble_sequence(zone),
            |_| 20.0, // Base dribble score
        )
        .hysteresis_margin(5.0)
        .description("Ball Carrier Decision")
        .build()
        .into()
}

/// Build shooting sequence
fn build_shoot_sequence() -> BehaviorNode {
    sequence_node()
        .add(
            face_position(find_optimal_shot_target)
                .with_ball()
                .description("Aim at goal".to_string())
                .build(),
        )
        .add(kick().description("Shoot!".to_string()).build())
        .description("Execute Shot")
        .build()
        .into()
}

/// Build passing sequence
fn build_pass_sequence() -> BehaviorNode {
    sequence_node()
        .add(
            face_position(find_best_pass_target)
                .with_ball()
                .description("Aim pass".to_string())
                .build(),
        )
        .add(kick().description("Pass!".to_string()).build())
        .description("Execute Pass")
        .build()
        .into()
}

/// Build dribbling sequence
fn build_dribble_sequence(zone: &str) -> BehaviorNode {
    go_to_position(calculate_striker_advance_position)
        .with_heading(move |s: &RobotSituation| {
            let goal_pos = s.get_opp_goal_position();
            let my_pos = s.player_data().position;
            Angle::between_points(goal_pos, my_pos)
        })
        .with_ball()
        .description(format!("Dribble in {}", zone))
        .build()
        .into()
}

fn get_kickoff_striker_position(s: &RobotSituation) -> Vector2 {
    let player_hash = s.player_id_hash() - 0.5;

    // Spread strikers across our half
    let spread_x = -500.0;
    let spread_y = if player_hash < 0.0 {
        -1200.0 - player_hash * 1000.0
    } else {
        1200.0 + player_hash * 1000.0
    };

    Vector2::new(spread_x, spread_y)
}

pub fn calculate_striker_advance_position(s: &RobotSituation) -> Vector2 {
    // Find optimal shot target
    let shot_target = find_optimal_shot_target(s);

    // Advance toward shot target, not just goal center
    let advance_direction = (shot_target - s.position()).normalize();
    let advance_distance = 1000.0;

    let target_pos = s.position() + advance_direction * advance_distance;

    // Ensure we stay in field
    s.constrain_to_field(target_pos)
}

fn find_optimal_striker_position(s: &RobotSituation, zone: &str) -> Vector2 {
    if let Some(field) = &s.world.field_geom {
        let half_width = field.field_width / 2.0;
        let third_height = field.field_width / 3.0;

        let y_center = match zone {
            "top" => half_width - third_height / 2.0,
            "bottom" => -half_width + third_height / 2.0,
            _ => 0.0, // middle
        };

        // Position in attacking third
        let x = field.field_length / 4.0;
        Vector2::new(x, y_center)
    } else {
        // Default positions
        match zone {
            "top" => Vector2::new(2000.0, 2000.0),
            "bottom" => Vector2::new(2000.0, -2000.0),
            _ => Vector2::new(2000.0, 0.0),
        }
    }
}

fn score_for_zone(s: &RobotSituation, zone: &str) -> f64 {
    // Base score with some randomization for diversity
    let hash = s.player_id_hash();
    let base_score = 50.0 + hash * 20.0;

    // Prefer zones with fewer opponents
    let zone_center = match zone {
        "top" => Vector2::new(2000.0, 2000.0),
        "bottom" => Vector2::new(2000.0, -2000.0),
        _ => Vector2::new(2000.0, 0.0),
    };

    let dist_score = (s.player_data().position - zone_center).norm() / 10.0;

    let opponents_in_zone = s.get_opp_players_within_radius(zone_center, 1500.0).len();

    let congestion_penalty = opponents_in_zone as f64 * 10.0;

    base_score - congestion_penalty - dist_score
}

/// Score for shooting vs passing decision
fn score_for_shooting(s: &RobotSituation) -> f64 {
    if !s.has_ball() {
        return 0.0;
    }

    let goal_pos = s.get_opp_goal_position();
    let my_pos = s.player_data().position;
    let goal_dist = (my_pos - goal_pos).norm();

    // Base score from distance (closer = better)
    let distance_score = (4000.0 - goal_dist.min(4000.0)) / 40.0;

    // Simple shot quality estimate
    let shot_quality = if has_clear_shot(s) { 1.0 } else { 0.3 };

    distance_score * shot_quality
}

/// Score for passing
fn score_for_passing(s: &RobotSituation) -> f64 {
    if !s.has_ball() || !can_pass_to_teammate(s) {
        return 0.0;
    }

    // Base score for having a pass option
    let mut score = 40.0;

    // Bonus if under pressure
    if under_pressure(s) {
        score += 30.0;
    }

    score
}

fn should_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return false;
    }

    let Some(ball) = s.world.ball.as_ref() else {
        return false;
    };

    let closest_opponent_dist = s
        .world
        .opp_players
        .iter()
        .map(|p| (ball.position.xy() - p.position).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap());

    ball.position.x > -1000.0
        && ball.velocity.xy().norm() < 500.0
        && closest_opponent_dist.map(|d| d > 1000.0).unwrap_or(true)
}
