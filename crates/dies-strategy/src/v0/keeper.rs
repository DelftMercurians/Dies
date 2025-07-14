use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn build_goalkeeper_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Penalty mode behavior
            guard_node()
                .condition(|s| s.is_penalty_state())
                .then(
                    go_to_position(|s: &RobotSituation| s.get_own_goal_position())
                        .description("Penalty Defense")
                        .build(),
                )
                .description("Penalty mode")
                .build(),
        )
        .add(
            // Emergency ball clearing if ball is very close
            guard_node()
                .description("Ball in penalty area")
                .condition(|s| s.ball_in_own_penalty_area() && s.ball_speed() < 500.0)
                .then(
                    sequence_node()
                        .description("Clear ball")
                        .add(fetch_ball().description("Clear Ball".to_string()).build())
                        .add(
                            guard_node()
                                .description("Have ball?")
                                .condition(|s| s.has_ball())
                                .then(
                                    sequence_node()
                                        .description("Execute Clear")
                                        .add(
                                            face_position(|s: &RobotSituation| {
                                                s.get_field_center()
                                            })
                                            .description("Face Field".to_string())
                                            .build(),
                                        )
                                        .add(kick().description("Clear!".to_string()).build())
                                        .build(),
                                )
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .add(
            // Normal goalkeeper behavior
            continuous("arc position")
                .position(calculate_arc_position)
                .heading(get_goalkeeper_heading)
                .build(),
        )
        .description("Goalkeeper")
        .build()
        .into()
}

fn calculate_arc_position(s: &RobotSituation) -> Vector2 {
    let geom = s.world.field_geom.clone().unwrap();
    let goal_pos = s.get_own_goal_position();

    // Define the three points of the shallow arc
    let left_point = Vector2::new(goal_pos.x, -geom.goal_width / 2.0);
    let right_point = Vector2::new(goal_pos.x, geom.goal_width / 2.0);
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
    if angle_diff > std::f64::consts::PI {
        angle_diff -= 2.0 * std::f64::consts::PI;
    } else if angle_diff < -std::f64::consts::PI {
        angle_diff += 2.0 * std::f64::consts::PI;
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

    // Negative distance to ball (closer is better)
    let ball_distance = (ball_pos - position).norm();
    score -= ball_distance / 3.0;

    // Score based on distance to ball velocity line (if ball is moving)
    if ball_vel.norm() > 100.0 {
        // Only consider if ball is moving
        let vel_direction = ball_vel.normalize();
        let ball_to_pos = position - ball_pos;

        // Project position onto ball velocity line
        let projection = ball_to_pos.dot(&vel_direction);
        let perpendicular_dist = (ball_to_pos - projection * vel_direction).norm();

        // Negative distance to velocity line (closer is better)
        score -= perpendicular_dist * 2.0;

        // Bonus for being ahead of the ball in the velocity direction
        if projection > 0.0 {
            score += projection / 2.0;
        }
    }

    score
}

fn get_goalkeeper_heading(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let goal_pos = s.get_own_goal_position();
        let ball_pos = ball.position.xy();
        Angle::between_points(ball_pos, goal_pos)
    } else {
        Angle::from_radians(0.0)
    }
}
