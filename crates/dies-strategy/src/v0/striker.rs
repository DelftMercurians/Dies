use dies_core::{Angle, GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{
    fetch_and_shoot, fetch_and_shoot_with_prep, find_best_receiver_target, get_heading_toward_ball,
};

pub fn build_striker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Handle kickoff positioning - stay on our side
            guard_node()
                .description("Kickoff positioning")
                .condition(|s| {
                    s.game_state_is_one_of(&[GameState::PrepareKickoff, GameState::Kickoff])
                })
                .then(
                    go_to_position(get_kickoff_striker_position)
                        .with_heading(get_heading_toward_ball)
                        .description("Kickoff positioning")
                        .build(),
                )
                .build(),
        )
        .add(try_receive())
        .add(
            // Pickup ball if we can
            committing_guard_node()
                .description("Ball pickup opportunity")
                .when(|s| should_pickup_ball(s)) // TODO: make sure that
                // the closest robots gets it; on second thought, this is probably not
                // implementable -> the roles are not available yet during the role assigment. so maybe not.
                .until(|_| false)
                .commit_to(
                    semaphore_node()
                        .do_then(fetch_and_shoot_with_prep())
                        .semaphore_id("striker_ball_pickup".to_string())
                        .max_entry(1)
                        .build(),
                )
                .build(),
        )
        .add(
            stateful_continuous("zoning")
                .with_stateful_position(|s, last_pos| {
                    let (target, score) = find_best_receiver_target(s, last_pos.copied());
                    (target, Some(target))
                })
                .build(),
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

/// Build striker behavior within a specific zone
fn build_striker_in_zone(zone: &str) -> BehaviorNode {
    let zone_owned = zone.to_string();
    let zone_owned_clone = zone_owned.clone();
    go_to_position(Argument::callback(move |s| {
        find_optimal_striker_position(s, &zone_owned_clone)
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
    .description(format!("Position in {}", zone_owned))
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

fn should_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return false;
    }
    if !s.can_touch_ball() {
        return false;
    }

    let Some(ball) = s.world.ball.as_ref() else {
        return false;
    };

    // am i closest striker
    let strikers = s
        .world
        .own_players
        .iter()
        .filter(|p| {
            s.role_assignments
                .get(&p.id)
                .cloned()
                .unwrap_or_default()
                .contains("striker")
        })
        .min_by(|a, b| {
            let a_dist = (ball.position.xy() - a.position).norm();
            let b_dist = (ball.position.xy() - b.position).norm();
            a_dist
                .partial_cmp(&b_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    let closest_striker = strikers.map(|p| p.id == s.player_id).unwrap_or(false);

    let closest_opponent_dist = s
        .world
        .opp_players
        .iter()
        .map(|p| (ball.position.xy() - p.position).norm())
        .min_by(|a, b| a.partial_cmp(b).unwrap());

    ball.position.x > 80.0
        && closest_opponent_dist.map(|d| d > 300.0).unwrap_or(true)
        && closest_striker
}
