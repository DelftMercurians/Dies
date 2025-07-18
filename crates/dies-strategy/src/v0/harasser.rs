use dies_core::{distance_to_line, GameState, Vector2, PLAYER_RADIUS};
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
            continuous("primary harasser")
                .position(Argument::callback(move |s| {
                    calculate_primary_harasser_position(s)
                }))
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

    if let (Some(field), Some(ball)) = (&s.world.field_geom, &s.world.ball) {
        let ball_pos = ball.position.xy();

        // If ball is in our penalty area, we don't want to pickup the ball
        let margin = 150.0;
        let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
        let y_bound = field.penalty_area_width / 2.0 - margin;
        if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
            return false;
        }

        // Only consider if ball is on our half
        if ball_pos.x >= 70.0 {
            return false;
        }

        let our_dist = s.distance_to_position(ball_pos);
        let closest_opp_dist = s.distance_of_closest_opp_player_to_ball();
        let on_the_line_to_goal =
            distance_to_line(s.ball_position(), s.get_own_goal_position(), s.position()) < 50.0;
        if our_dist > closest_opp_dist && !on_the_line_to_goal {
            return false;
        }

        // Check if ball is not moving fast
        let ball_velocity = ball.velocity.xy();
        if ball_velocity.norm() > 1000.0 {
            return false;
        }

        true
    } else {
        false
    }
}

fn should_cancel_pickup_ball(s: &RobotSituation) -> bool {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        if let Some(field) = &s.world.field_geom {
            let margin = 150.0;
            let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
            let y_bound = field.penalty_area_width / 2.0 - margin;
            if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
                return true;
            }
        }
    }

    s.ball_position().x > 100.0 || s.distance_of_closest_opp_player_to_ball() < 150.0
}

fn calculate_primary_harasser_position(s: &RobotSituation) -> Vector2 {
    let ball_pos = s.ball_position();
    let own_goal = s.get_own_goal_position();

    // start by checking if the ball is in our defence area, if so -> get the fuck out of the way
    if let Some(field) = &s.world.field_geom {
        let margin = 150.0; // at what distance the ball in penalty area is safe from attackers
        let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
        let y_bound = field.penalty_area_width / 2.0 - margin;
        if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
            // in this case we actually want to just stay somewhere non-blocking
            return Vector2::new(-3100.0, -1000.0);
        }
    }

    // Default: position on line from ball towards our goal
    // Check if the closest player is on of our own by a margin of 0.5
    let closest_own_striker = s
        .get_players_with_role("striker")
        .into_iter()
        .min_by_key(|p| (p.position - ball_pos).norm() as i64);
    let closest_opp = s.get_closest_opp_player_to_ball();
    let closest_opp_dist = closest_opp
        .map(|opp| (opp.position - ball_pos).norm())
        .unwrap_or(f64::INFINITY);
    let closest_own_dist = closest_own_striker
        .as_ref()
        .map(|own| (own.position - ball_pos).norm())
        .unwrap_or(f64::INFINITY);
    let harass_distance = if closest_opp_dist > closest_own_dist + 500.0
        && closest_own_striker
            .map(|p| p.id != s.player_id)
            .unwrap_or(true)
    {
        // Get out of the way
        1000.0
    } else {
        300.0
    };
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
