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
                .condition(|s| s.ball_in_own_penalty_area() && s.distance_to_ball() < 1000.0)
                .then(
                    sequence_node()
                        .add(fetch_ball().description("Clear Ball".to_string()).build())
                        .add(
                            guard_node()
                                .condition(|s| s.has_ball())
                                .then(
                                    sequence_node()
                                        .add(
                                            face_position(|s: &RobotSituation| {
                                                s.get_field_center()
                                            })
                                            .description("Face Field".to_string())
                                            .build(),
                                        )
                                        .add(kick().description("Clear!".to_string()).build())
                                        .description("Execute Clear")
                                        .build(),
                                )
                                .description("Have ball?")
                                .build(),
                        )
                        .description("Emergency Clear")
                        .build(),
                )
                .description("Ball in penalty area")
                .build(),
        )
        .add(
            // Normal goalkeeper behavior
            go_to_position(calculate_arc_position)
                .with_heading(get_goalkeeper_heading)
                .description("Guard Goal (arc)")
                .build(),
        )
        .description("Goalkeeper")
        .build()
        .into()
}

fn calculate_goalkeeper_position(s: &RobotSituation) -> Vector2 {
    let geom = s.world.field_geom.clone().unwrap();
    let goal_line_start = Vector2::new(geom.field_length / 2.0, -geom.goal_width / 2.0);
    let goal_line_start = Vector2::new(geom.field_length / 2.0, geom.goal_width / 2.0);
    let goal_pos = s.get_own_goal_position();
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let direction = (ball_pos - goal_pos).normalize();
        goal_pos + direction * 800.0 // 800mm from goal
    } else {
        goal_pos + Vector2::new(500.0, 0.0) // Default position
    }
}

fn calculate_arc_position(s: &RobotSituation) -> Vector2 {
    let geom = s.world.field_geom.clone().unwrap();
    let goal_pos = s.get_own_goal_position();

    // Define the three points of the shallow arc
    let left_point = Vector2::new(goal_pos.x, -geom.goal_width / 2.0);
    let right_point = Vector2::new(goal_pos.x, geom.goal_width / 2.0);
    let forward_point = Vector2::new(goal_pos.x - 400.0, 0.0); // 40cm forward from goal center

    // Generate 50 sample points along the arc
    let mut arc_points = Vec::new();

    // Sample points along the arc connecting left -> forward -> right
    for i in 0..50 {
        let t = i as f64 / 49.0; // 0 to 1

        let point = if t <= 0.5 {
            // First half: left to forward
            let local_t = t * 2.0;
            interpolate_arc_point(left_point, forward_point, goal_pos, local_t)
        } else {
            // Second half: forward to right
            let local_t = (t - 0.5) * 2.0;
            interpolate_arc_point(forward_point, right_point, goal_pos, local_t)
        };

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

fn interpolate_arc_point(start: Vector2, end: Vector2, center: Vector2, t: f64) -> Vector2 {
    // Create a shallow arc by interpolating through a point slightly forward from the midpoint
    let midpoint = (start + end) * 0.5;
    let arc_center = midpoint + (center - midpoint).normalize() * 100.0; // 10cm forward arc

    // Use quadratic interpolation for smooth arc
    let p0 = start;
    let p1 = arc_center;
    let p2 = end;

    // Quadratic Bezier curve: (1-t)²P0 + 2(1-t)tP1 + t²P2
    let one_minus_t = 1.0 - t;
    p0 * (one_minus_t * one_minus_t) + p1 * (2.0 * one_minus_t * t) + p2 * (t * t)
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
    if ball_vel.norm() > 100.0 { // Only consider if ball is moving
        let vel_direction = ball_vel.normalize();
        let ball_to_pos = position - ball_pos;

        // Project position onto ball velocity line
        let projection = ball_to_pos.dot(&vel_direction);
        let perpendicular_dist = (ball_to_pos - projection * vel_direction).norm();

        // Negative distance to velocity line (closer is better)
        score -= perpendicular_dist;

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
