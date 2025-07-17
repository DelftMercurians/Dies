use core::f64;

use dies_core::{GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{fetch_and_shoot_with_prep, get_heading_toward_ball};

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
        .add(
            // Pickup ball if we can
            committing_guard_node()
                .description("Ball pickup opportunity")
                .when(|s| should_pickup_ball(s))
                .until(|s| s.position().x < -100.0)
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
            scoring_select_node()
                .description("Striker tactics")
                .hysteresis_margin(0.1)
                .add_child(
                    semaphore_node()
                        .semaphore_id("striker_harasser".to_string())
                        .max_entry(1)
                        .do_then(
                            continuous("Harassing opponent")
                                .position(Argument::callback(striker_harassing_position))
                                .build(),
                        )
                        .build(),
                    striker_harassing_score,
                )
                .add_child(
                    stateful_continuous("Ball tracking zoning")
                        .with_stateful_position(|s, last_pos| {
                            let target = striker_ball_tracking_position(s, last_pos.copied());
                            (target, Some(target))
                        })
                        .build(),
                    |_| 40.0,
                )
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

    score
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

fn should_pickup_ball(s: &RobotSituation) -> bool {
    return true;
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
    let my_dist = s.distance_to_ball();
    let d = closest_opponent_dist.unwrap_or(f64::INFINITY) - my_dist;

    ball.position.x > -80.0
        && (closest_opponent_dist.map(|d| d > 300.0).unwrap_or(true))
        && closest_striker
}

fn striker_harassing_score(s: &RobotSituation) -> f64 {
    if s.game_state_is_not(GameState::Run) {
        return 0.0;
    }

    let ball_pos = s.ball_position();

    // Only harass if ball is on opponent side
    if ball_pos.x <= 0.0 {
        return 0.0;
    }

    // Find closest opponent to ball
    if let Some(closest_opp) = s.get_closest_opp_player_to_ball() {
        let opp_to_ball_dist = (closest_opp.position - ball_pos).norm();

        // Only harass if opponent is close to ball
        if opp_to_ball_dist > 500.0 {
            return 0.0;
        }

        let mut score = 60.0;

        // Prefer being close to the opponent
        let my_dist_to_opp = s.distance_to_position(closest_opp.position);
        score += (1500.0 - my_dist_to_opp.min(1500.0)) / 30.0;

        return score;
    }

    0.0
}

fn striker_harassing_position(s: &RobotSituation) -> Vector2 {
    // Check if we need to move out of the way of a teammate's shot
    if let Some(avoid_pos) = get_out_of_shot_position(s) {
        return avoid_pos;
    }

    let ball_pos = s.ball_position();

    if let Some(closest_opp) = s.get_closest_opp_player_to_ball() {
        let opp_to_ball_dist = (closest_opp.position - ball_pos).norm();

        if opp_to_ball_dist < 200.0 {
            // Ball is very close to opponent: position directly in front using opponent's yaw
            let harass_distance = 350.0;
            let opp_yaw = closest_opp.yaw;
            let facing_vec = opp_yaw.to_vector();
            let target_pos = closest_opp.position + facing_vec * harass_distance;

            let constrained_pos = s.constrain_to_field(target_pos);
            return Vector2::new(constrained_pos.x.max(50.0), constrained_pos.y);
        } else {
            // Position between opponent and ball, maintaining distance
            let harass_distance = 350.0;
            let to_ball = (ball_pos - closest_opp.position).normalize();
            let target_pos = closest_opp.position + to_ball * harass_distance;

            // Ensure we stay on opponent side
            let constrained_pos = s.constrain_to_field(target_pos);
            return Vector2::new(constrained_pos.x.max(50.0), constrained_pos.y);
        }
    }

    // Fallback to ball position
    Vector2::new(ball_pos.x.max(50.0), ball_pos.y)
}

fn striker_ball_tracking_position(s: &RobotSituation, last_pos: Option<Vector2>) -> Vector2 {
    // Check if we need to move out of the way of a teammate's shot
    if let Some(avoid_pos) = get_out_of_shot_position(s) {
        return avoid_pos;
    }

    let ball_pos = s.ball_position();
    let goal_pos = s.get_opp_goal_position();

    // Start with ball position as base target
    let mut target_x = ball_pos.x * 0.7 + goal_pos.x * 0.3; // Gravitate towards ball but stay closer to goal
    let mut target_y = ball_pos.y * 0.6 + goal_pos.y * 0.4;

    // Apply repulsion from other strikers
    for player in s.get_players_with_role("striker") {
        if player.id == s.player_id {
            continue;
        }

        let to_player = Vector2::new(target_x, target_y) - player.position;
        let dist = to_player.norm() + f64::EPSILON;

        if dist < 800.0 {
            let repulsion_strength = (800.0 - dist) / 800.0 * 600.0;
            let repulsion = to_player.normalize() * repulsion_strength;
            target_x += repulsion.x;
            target_y += repulsion.y;
        }
    }

    let mut target = Vector2::new(target_x, target_y);

    // Constrain to opponent half
    target.x = target.x.max(50.0);

    // Avoid defense area
    if let Some(field) = &s.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        // Check opponent penalty area
        if target.x >= half_length - field.penalty_area_depth
            && target.y >= -half_penalty_width
            && target.y <= half_penalty_width
        {
            // Push out of penalty area
            if target.y.abs() < half_penalty_width {
                target.y = if target.y >= 0.0 {
                    half_penalty_width + 100.0
                } else {
                    -half_penalty_width - 100.0
                };
            }
        }
    }

    // Apply stability - prefer staying near last position
    if let Some(last_pos) = last_pos {
        let distance_to_last = (target - last_pos).norm();
        if distance_to_last < 100.0 {
            // Small movements - interpolate for stability
            target = last_pos * 0.7 + target * 0.3;
        }
    }

    // Final field constraint
    s.constrain_to_field(target)
}

fn get_out_of_shot_position(s: &RobotSituation) -> Option<Vector2> {
    // Find teammate with ball who is aiming at goal
    let ball_carrier = find_ball_carrier_aiming_at_goal(s)?;

    // Check if we're blocking the shot
    if !is_blocking_shot_to_goal(s, &ball_carrier) {
        return None;
    }

    // Calculate position to move out of the way
    Some(calculate_avoidance_position(s, &ball_carrier))
}

fn find_ball_carrier_aiming_at_goal(s: &RobotSituation) -> Option<dies_core::PlayerData> {
    let goal_pos = s.get_opp_goal_position();

    for player in &s.world.own_players {
        if player.id == s.player_id {
            continue;
        }

        // Check if player has the ball
        if !player.breakbeam_ball_detected {
            continue;
        }

        // Check if player is facing towards goal (within 45 degrees)
        let to_goal = (goal_pos - player.position).normalize();
        let player_facing = player.yaw.to_vector();
        let angle_diff = to_goal.dot(&player_facing).acos();

        if angle_diff < 0.785 {
            // 45 degrees in radians
            return Some(player.clone());
        }
    }

    None
}

fn is_blocking_shot_to_goal(s: &RobotSituation, ball_carrier: &dies_core::PlayerData) -> bool {
    let goal_pos = s.get_opp_goal_position();
    let my_pos = s.position();
    let carrier_pos = ball_carrier.position;

    // Calculate distance from my position to the shot line
    let shot_direction = (goal_pos - carrier_pos).normalize();
    let to_me = my_pos - carrier_pos;

    // Project my position onto the shot line
    let projection_length = to_me.dot(&shot_direction);

    // Only consider if I'm between carrier and goal
    if projection_length <= 0.0 || projection_length >= (goal_pos - carrier_pos).norm() {
        return false;
    }

    // Calculate perpendicular distance to shot line
    let projected_point = carrier_pos + shot_direction * projection_length;
    let distance_to_line = (my_pos - projected_point).norm();

    // Consider blocking if within robot radius + margin
    distance_to_line < 150.0 // Robot radius + some margin
}

fn calculate_avoidance_position(
    s: &RobotSituation,
    ball_carrier: &dies_core::PlayerData,
) -> Vector2 {
    let goal_pos = s.get_opp_goal_position();
    let my_pos = s.position();
    let carrier_pos = ball_carrier.position;

    // Calculate shot direction
    let shot_direction = (goal_pos - carrier_pos).normalize();
    let perpendicular = Vector2::new(-shot_direction.y, shot_direction.x);

    // Move to the side that's closer to my current position
    let to_me = my_pos - carrier_pos;
    let side = if to_me.dot(&perpendicular) > 0.0 {
        1.0
    } else {
        -1.0
    };

    // Calculate avoidance position
    let avoidance_distance = 200.0; // Distance to move away from shot line
    let along_shot = to_me.dot(&shot_direction).max(200.0); // Stay ahead of carrier

    let avoid_pos =
        carrier_pos + shot_direction * along_shot + perpendicular * side * avoidance_distance;

    // Ensure we stay on opponent side and out of penalty area
    let mut final_pos = Vector2::new(avoid_pos.x.max(50.0), avoid_pos.y);

    // Avoid opponent penalty area
    if let Some(field) = &s.world.field_geom {
        let half_length = field.field_length / 2.0;
        let half_penalty_width = field.penalty_area_width / 2.0;

        if final_pos.x >= half_length - field.penalty_area_depth
            && final_pos.y.abs() <= half_penalty_width
        {
            // Push out of penalty area
            final_pos.y = if final_pos.y >= 0.0 {
                half_penalty_width + 100.0
            } else {
                -half_penalty_width - 100.0
            };
        }
    }

    s.constrain_to_field(final_pos)
}
