use std::f64::consts::PI;

use dies_core::{Angle, Vector2};
use dies_strategy_protocol::GameState;

use crate::bt::*;
use crate::helpers::nearest_player_distance_along_direction;

pub fn build_goalkeeper_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Emergency ball clearing if a slow ball is in our penalty area.
            guard_node()
                .description("Ball in defense area")
                .condition(|s| s.ball_in_own_penalty_area() && s.ball_speed() < 500.0)
                .then(
                    fetch_ball_with_preshoot()
                        .with_avoid_ball_care(1.0)
                        .with_override_target(find_clear_exit_target)
                        .description("Clear Ball")
                        .build(),
                )
                .build(),
        )
        .add(
            // Normal goalkeeper arc positioning.
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

/// Find an unobstructed direction to clear the ball out of the penalty area.
fn find_clear_exit_target(s: &RobotSituation) -> ShootTarget {
    let Some(ball) = &s.world.ball else {
        return ShootTarget::Goal(s.get_opp_goal_position());
    };
    let ball_pos = ball.position;

    let num_samples = 36;
    let mut best_target = s.get_opp_goal_position();
    let mut best_score = f64::NEG_INFINITY;

    let min_angle = 15.0f64.to_radians();
    let total_angle = PI - min_angle;
    for i in 0..num_samples {
        let angle_rad = (i as f64 / num_samples as f64) * total_angle - (total_angle / 2.0);
        let direction = Angle::from_radians(angle_rad);

        let direction_vector = direction.to_vector();
        let target_distance = 2000.0;
        let target_pos = ball_pos + direction_vector * target_distance;

        let score = score_exit_direction(s, direction, target_pos);
        if score > best_score {
            best_score = score;
            best_target = target_pos;
        }
    }

    ShootTarget::Goal(best_target)
}

fn score_exit_direction(s: &RobotSituation, direction: Angle, target_pos: Vector2) -> f64 {
    let own_goal = s.get_own_goal_position();
    let ball_pos = s.ball_position();
    let to_target = (target_pos - ball_pos).normalize();
    let to_goal = (own_goal - ball_pos).normalize();

    // Never clear towards our own goal.
    if to_target.dot(&to_goal) > 0.5 {
        return f64::NEG_INFINITY;
    }

    let mut score = 0.0;

    let opponent_distance =
        nearest_player_distance_along_direction(s, ball_pos, direction.to_vector());
    let opponent_score = if opponent_distance > 1000.0 {
        1.0
    } else if opponent_distance > 500.0 {
        0.7
    } else if opponent_distance > 300.0 {
        0.4
    } else {
        0.1
    };
    score += opponent_score * 5.0;

    let away_from_goal_score = 1.0 - to_target.dot(&to_goal);
    score += away_from_goal_score;

    score
}

fn calculate_arc_position(s: &RobotSituation) -> Vector2 {
    let geom = s.field();
    let goal_pos = s.get_own_goal_position();

    let left_point = Vector2::new(goal_pos.x + 50.0, -geom.goal_width / 2.0);
    let right_point = Vector2::new(goal_pos.x + 50.0, geom.goal_width / 2.0);
    let forward_point = Vector2::new(goal_pos.x + 400.0, 0.0);

    let (circle_center, circle_radius) =
        calculate_circumcircle(left_point, right_point, forward_point);

    let mut arc_points = Vec::new();
    let start_angle = (left_point - circle_center)
        .y
        .atan2((left_point - circle_center).x);
    let end_angle = (right_point - circle_center)
        .y
        .atan2((left_point - circle_center).x);

    let mut angle_diff = end_angle - start_angle;
    if angle_diff > PI {
        angle_diff -= 2.0 * PI;
    } else if angle_diff < -PI {
        angle_diff += 2.0 * PI;
    }

    for i in 0..50 {
        let t = i as f64 / 49.0;
        let angle = start_angle + angle_diff * t;
        let point =
            circle_center + Vector2::new(circle_radius * angle.cos(), circle_radius * angle.sin());
        arc_points.push(point);
    }

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
    let d = 2.0 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y));

    if d.abs() < 1e-10 {
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

    let ball_pos = ball.position;
    let ball_vel = ball.velocity;
    let mut score = 0.0;

    let goal_pos = s.get_own_goal_position();
    let goal_distance = (position - goal_pos).norm();
    score -= goal_distance * 1.2;

    let ball_distance = (ball_pos + ball_vel * 0.3 - position).norm();
    score -= ball_distance;

    if ball_vel.norm() > 50.0 {
        let vel_direction = ball_vel.normalize();
        let ball_to_pos = position - ball_pos;

        let projection = ball_to_pos.dot(&vel_direction);
        let parallel_dist = projection * vel_direction;
        let perpendicular_dist = (ball_to_pos - parallel_dist).norm().clamp(-2000.0, 2000.0);

        let time_to_intersect = if projection <= 0.0 {
            20.0
        } else {
            parallel_dist.norm() / ball_vel.x.abs()
        };

        score -= perpendicular_dist * 10.0 / time_to_intersect.max(0.5);
    }

    score
}

fn get_goalkeeper_heading(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        Angle::between_points(s.get_own_goal_position(), ball.position)
    } else {
        Angle::from_radians(0.0)
    }
}
