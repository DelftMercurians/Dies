use std::f64::consts::{FRAC_PI_2, PI};
use std::fmt::format;

use dies_core::{Angle, GameState, Vector2};
use dies_executor::control::{PassingStore, ShootTarget};
use dies_executor::{
    behavior_tree_api::*, find_nearest_opponent_distance_along_direction,
    find_nearest_player_distance_along_direction,
};

pub fn build_goalkeeper_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        // .add(
        //     // Penalty mode behavior
        //     guard_node()
        //         .condition(|s| s.is_penalty_state())
        //         .then(
        //             go_to_position(|s: &RobotSituation| s.get_own_goal_position())
        //                 .description("Penalty Defense")
        //                 .build(),
        //         )
        //         .description("Penalty mode")
        //         .build(),
        // )
        .add(
            // Emergency ball clearing if ball is very close
            guard_node()
                .description("Ball in defense area")
                .condition(|s| s.ball_in_own_penalty_area() && s.ball_speed() < 500.0)
                .then(
                    fetch_ball_with_preshoot()
                        .with_avoid_ball_care(1.0)
                        .with_override_target(find_clear_exit_target)
                        .description("Clear Ball".to_string())
                        .build(),
                )
                .build(),
        )
        .add(
            // Normal goalkeeper behavior
            continuous("arc position")
                .position(calculate_arc_position)
                .heading(get_goalkeeper_heading)
                .aggressiveness(6.0)
                .carefullness(-40.0)
                .build(),
        )
        .description("Goalkeeper")
        .build()
        .into()
}

/// Find an unobstructed direction to clear the ball out of the penalty area
fn find_clear_exit_target(s: &RobotSituation) -> ShootTarget {
    let Some(ball) = &s.world.ball else {
        return ShootTarget::Goal(s.get_opp_goal_position());
    };

    let ball_pos = ball.position.xy();
    let passing_store = PassingStore::from(s);

    // Sample different angles to find the clearest exit
    let num_samples = 36;
    let mut best_target = s.get_opp_goal_position();
    let mut best_score = f64::NEG_INFINITY;

    let min_angle = 15.0f64.to_radians();
    let total_angle = PI - min_angle;
    for i in 0..num_samples {
        let angle_rad = (i as f64 / num_samples as f64) * total_angle - (total_angle / 2.0);
        let direction = Angle::from_radians(angle_rad);

        // Project this direction to a target position well outside the penalty area
        let direction_vector = direction.to_vector();
        let target_distance = 2000.0; // 2 meters away from ball
        let target_pos = ball_pos + direction_vector * target_distance;

        s.team_context
            .debug_cross(format!("exit_{}", i), target_pos);

        // Score this direction
        let score = score_exit_direction(s, &passing_store, direction, target_pos);

        if score > best_score {
            best_score = score;
            best_target = target_pos;
        }
    }
    s.team_context.debug_cross_colored(
        format!("exit_{}", 0),
        best_target,
        dies_core::DebugColor::Blue,
    );

    ShootTarget::Goal(best_target)
}

/// Score an exit direction based on how clear it is and how well it exits the penalty area
fn score_exit_direction(
    s: &RobotSituation,
    passing_store: &PassingStore,
    direction: Angle,
    target_pos: Vector2,
) -> f64 {
    // Safety check: Never allow directions towards own goal
    let own_goal = s.get_own_goal_position();
    let ball_pos = s.ball_position();
    let to_target = (target_pos - ball_pos).normalize();
    let to_goal = (own_goal - ball_pos).normalize();

    // If direction is towards own goal (dot product > 0.5, meaning angle < 60°), reject it
    if to_target.dot(&to_goal) > 0.5 {
        return f64::NEG_INFINITY;
    }

    let mut score = 0.0;

    // Factor 1: Distance to nearest opponent in this direction (higher is better)
    let opponent_distance = find_nearest_player_distance_along_direction(passing_store, direction);
    let opponent_score = if opponent_distance > 1000.0 {
        1.0
    } else if opponent_distance > 500.0 {
        0.7
    } else if opponent_distance > 300.0 {
        0.4
    } else {
        0.1
    };
    score += opponent_score * 5.0; // High weight for opponent clearance

    // Factor 2: Prefer directions that lead away from our goal
    let own_goal = s.get_own_goal_position();
    let ball_pos = s.ball_position();
    let to_target = (target_pos - ball_pos).normalize();
    let to_goal = (own_goal - ball_pos).normalize();
    let away_from_goal_score = 1.0 - to_target.dot(&to_goal); // Range 0-2, higher when pointing away
    score += away_from_goal_score;

    score
}

fn calculate_arc_position(s: &RobotSituation) -> Vector2 {
    let geom = s.world.field_geom.clone().unwrap();
    let goal_pos = s.get_own_goal_position();

    // Define the three points of the shallow arc
    let left_point = Vector2::new(goal_pos.x + 50.0, -geom.goal_width / 2.0);
    let right_point = Vector2::new(goal_pos.x + 50.0, geom.goal_width / 2.0);
    let forward_point = Vector2::new(goal_pos.x + 400.0, 0.0); // 40cm forward from goal center

    // Calculate circle center and radius that passes through all three points
    let (circle_center, circle_radius) =
        calculate_circumcircle(left_point, right_point, forward_point);

    // Generate 50 sample points uniformly around the circle arc
    let mut arc_points = Vec::new();

    // Calculate start and end angles for the arc
    let start_angle = (left_point - circle_center)
        .y
        .atan2((left_point - circle_center).x);
    let end_angle = (right_point - circle_center)
        .y
        .atan2((left_point - circle_center).x);

    // Determine the shorter arc direction
    let mut angle_diff = end_angle - start_angle;
    if angle_diff > PI {
        angle_diff -= 2.0 * PI;
    } else if angle_diff < -PI {
        angle_diff += 2.0 * PI;
    }

    // Sample points uniformly by angle
    for i in 0..50 {
        let t = i as f64 / 49.0; // 0 to 1
        let angle = start_angle + angle_diff * t;

        let point =
            circle_center + Vector2::new(circle_radius * angle.cos(), circle_radius * angle.sin());

        arc_points.push(point);
    }

    // Score each point and find the best one
    let mut best_point = arc_points[0];
    let mut best_score = score_arc_position(s, best_point);

    for &point in &arc_points[1..] {
        let score = score_arc_position(s, point);
        if score > best_score {
            best_score = score;
            best_point = point;
        }
    }

    if s.game_state_is_one_of(&[
        GameState::Penalty,
        GameState::PreparePenalty,
        GameState::PenaltyRun,
    ]) {
        let mut best_point = best_point;
        best_point.x = s.get_own_goal_position().x;
        return best_point;
    }

    best_point
}

fn calculate_circumcircle(p1: Vector2, p2: Vector2, p3: Vector2) -> (Vector2, f64) {
    // Calculate the circumcenter and circumradius of the triangle formed by three points
    let d = 2.0 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));

    // Handle degenerate case (collinear points)
    if d.abs() < 1e-10 {
        // Fallback to midpoint between extremes
        let center = (p1 + p3) * 0.5;
        let radius = (p1 - center).norm();
        return (center, radius);
    }

    let ux = (p1.x * p1.x + p1.y * p1.y) * (p2.y - p3.y)
        + (p2.x * p2.x + p2.y * p2.y) * (p3.y - p1.y)
        + (p3.x * p3.x + p3.y * p3.y) * (p1.y - p2.y);

    let uy = (p1.x * p1.x + p1.y * p1.y) * (p3.x - p2.x)
        + (p2.x * p2.x + p2.y * p2.y) * (p1.x - p3.x)
        + (p3.x * p3.x + p3.y * p3.y) * (p2.x - p1.x);

    let center = Vector2::new(ux / d, uy / d);
    let radius = (p1 - center).norm();

    (center, radius)
}

fn score_arc_position(s: &RobotSituation, position: Vector2) -> f64 {
    let Some(ball) = &s.world.ball else {
        return 0.0;
    };

    let ball_pos = ball.position.xy();
    let ball_vel = ball.velocity.xy();
    let mut score = 0.0;

    // be close to the classical position
    let goal_pos = s.get_own_goal_position();
    let goal_distance = (position - goal_pos).norm();
    score -= goal_distance * 1.2;

    // Negative distance to ball (closer is better)

    let ball_distance = (ball_pos + ball_vel * 0.3 - position).norm();
    score -= ball_distance;

    // Score based on distance to ball velocity line (if ball is moving)
    if ball_vel.norm() > 50.0 {
        // Only consider if ball is moving
        let vel_direction = ball_vel.normalize();
        let ball_to_pos = position - ball_pos;

        // Project position onto ball velocity line
        let projection = ball_to_pos.dot(&vel_direction);
        let parallel_dist = projection * vel_direction;
        let perpendicular_dist = (ball_to_pos - parallel_dist).norm().clamp(-2000.0, 2000.0);

        let mut time_to_intersect = if projection <= 0.0 {
            20.0
        } else {
            parallel_dist.norm() / ball_vel.x.abs()
        };

        // Negative distance to velocity line (closer is better)
        score -= perpendicular_dist * 10.0 / time_to_intersect.max(0.5);
    }

    score
}

fn get_goalkeeper_heading(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let goal_pos = s.get_own_goal_position();
        let ball_pos = ball.position.xy();
        Angle::between_points(goal_pos, ball_pos)
    } else {
        Angle::from_radians(0.0)
    }
}
