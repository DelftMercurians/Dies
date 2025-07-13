use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{evaluate_ball_threat, get_defender_heading};

pub fn build_waller_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Normal wall positioning
            go_to_position(Argument::callback(|s| calculate_wall_position(s)))
                .with_heading(Argument::callback(get_defender_heading))
                .description("Wall position".to_string())
                .build(),
        )
        .description("Waller")
        .build()
        .into()
}

pub fn score_as_waller(s: &RobotSituation) -> f64 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let goal_pos = s.get_own_goal_position();

        // Base score for defender role
        let mut score = 40.0;

        // Higher score if ball is in central threatening position
        let ball_threat = evaluate_ball_threat(s);
        score += ball_threat * 30.0;

        // Higher score if already positioned between ball and goal
        let positioning_score = evaluate_waller_positioning(s, ball_pos, goal_pos);
        score += positioning_score * 25.0;

        // Lower score if too far from defensive area
        let goal_dist = s.distance_to_position(goal_pos);
        if goal_dist > 3000.0 {
            score -= 20.0;
        }

        score
    } else {
        0.0
    }
}

fn is_ball_threatening(s: &RobotSituation) -> bool {
    if let Some(ball) = &s.world.ball {
        let ball_vel = ball.velocity;
        let goal_pos = s.get_own_goal_position();
        let ball_pos = ball.position.xy();

        // Check if ball is moving toward our goal
        let ball_to_goal = (goal_pos - ball_pos).normalize();
        let vel_direction = ball_vel.normalize();

        let moving_toward_goal =
            ball_to_goal.x * vel_direction.x + ball_to_goal.y * vel_direction.y > 0.5;
        let in_our_half = ball_pos.x < 0.0;
        let moving_fast = ball_vel.norm() > 500.0;

        moving_toward_goal && in_our_half && moving_fast
    } else {
        false
    }
}

fn calculate_wall_position(s: &RobotSituation) -> Vector2 {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        find_best_boundary_position(s, ball_pos)
    } else {
        s.player_data().position
    }
}

fn get_defense_area_boundary(s: &RobotSituation) -> Option<DefenseAreaBoundary> {
    if let Some(field) = &s.world.field_geom {
        let goal_line_x = -field.field_length / 2.0;
        let penalty_line_x = goal_line_x + field.penalty_area_depth;
        let half_width = field.penalty_area_width / 2.0;

        Some(DefenseAreaBoundary {
            goal_line_x,
            penalty_line_x,
            top_y: half_width,
            bottom_y: -half_width,
        })
    } else {
        None
    }
}

struct DefenseAreaBoundary {
    goal_line_x: f64,
    penalty_line_x: f64,
    top_y: f64,
    bottom_y: f64,
}

impl DefenseAreaBoundary {
    fn get_boundary_segments(&self) -> Vec<(Vector2, Vector2)> {
        vec![
            // Top horizontal line
            (
                Vector2::new(self.goal_line_x, self.top_y),
                Vector2::new(self.penalty_line_x, self.top_y),
            ),
            // Right vertical line (penalty line)
            (
                Vector2::new(self.penalty_line_x, self.top_y),
                Vector2::new(self.penalty_line_x, self.bottom_y),
            ),
            // Bottom horizontal line
            (
                Vector2::new(self.penalty_line_x, self.bottom_y),
                Vector2::new(self.goal_line_x, self.bottom_y),
            ),
            // Left vertical line (goal line) - only the parts not covered by goal
            (
                Vector2::new(self.goal_line_x, self.bottom_y),
                Vector2::new(self.goal_line_x, self.top_y),
            ),
        ]
    }
}

fn find_best_boundary_position(s: &RobotSituation, ball_pos: Vector2) -> Vector2 {
    if let Some(boundary) = get_defense_area_boundary(s) {
        let mut wallers = s
            .role_assignments
            .iter()
            .filter(|(_, v)| *v == "waller")
            .collect::<Vec<_>>();
        wallers.sort_by_key(|(k, _)| *k);

        let waller_index = wallers
            .iter()
            .position(|(k, _)| **k == s.player_id)
            .unwrap_or(0);

        let goal_pos = s.get_own_goal_position();
        let ball_to_goal = goal_pos - ball_pos;

        // Find the best point on the boundary to intercept the ball-to-goal line
        let segments = boundary.get_boundary_segments();
        let mut center_position = Vector2::new(boundary.penalty_line_x, 0.0); // Default fallback
        let mut best_score = f64::NEG_INFINITY;

        for (start, end) in segments {
            if let Some(intercept) =
                line_intersection(ball_pos, ball_pos + ball_to_goal, start, end)
            {
                // Check if intercept is actually on the segment
                if is_point_on_segment(intercept, start, end) {
                    // Score based on how well this position blocks the threat
                    let score = evaluate_boundary_position(s, intercept, ball_pos, goal_pos);
                    if score > best_score {
                        best_score = score;
                        center_position = intercept;
                    }
                }
            }
        }

        // If no direct intercept found, find closest point on boundary to ball-goal line
        if best_score == f64::NEG_INFINITY {
            center_position = find_closest_boundary_point(s, ball_pos, goal_pos);
        }

        center_position // TODO: Place wallers smartly
    } else {
        // Fallback to old behavior if no field geometry
        s.player_data().position
    }
}

fn line_intersection(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2) -> Option<Vector2> {
    let denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if denom.abs() < 1e-10 {
        return None; // Lines are parallel
    }

    let t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    Some(Vector2::new(
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y),
    ))
}

fn is_point_on_segment(point: Vector2, start: Vector2, end: Vector2) -> bool {
    let cross_product =
        (point.y - start.y) * (end.x - start.x) - (point.x - start.x) * (end.y - start.y);
    if cross_product.abs() > 1e-6 {
        return false; // Point is not on the line
    }

    let dot_product =
        (point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y);
    let squared_length = (end.x - start.x).powi(2) + (end.y - start.y).powi(2);

    dot_product >= 0.0 && dot_product <= squared_length
}

fn evaluate_boundary_position(
    s: &RobotSituation,
    pos: Vector2,
    ball_pos: Vector2,
    goal_pos: Vector2,
) -> f64 {
    // Higher score for positions that better block the ball-to-goal line
    let ball_to_goal = goal_pos - ball_pos;
    let ball_to_pos = pos - ball_pos;

    // Project position onto ball-goal line to see how well it blocks
    let projection = ball_to_pos.dot(&ball_to_goal.normalize());
    let perpendicular_dist = (ball_to_pos - ball_to_goal.normalize() * projection).norm();

    // Score based on being on the line and distance from ball
    let blocking_score = 1000.0 / (1.0 + perpendicular_dist);
    let distance_score = projection.max(0.0) / ball_to_goal.norm();

    blocking_score + distance_score * 50.0
}

fn find_closest_boundary_point(
    s: &RobotSituation,
    ball_pos: Vector2,
    goal_pos: Vector2,
) -> Vector2 {
    if let Some(boundary) = get_defense_area_boundary(s) {
        let ball_to_goal = goal_pos - ball_pos;
        let segments = boundary.get_boundary_segments();

        let mut closest_point = Vector2::new(boundary.penalty_line_x, 0.0);
        let mut min_distance = f64::INFINITY;

        for (start, end) in segments {
            let point = closest_point_on_segment(ball_pos + ball_to_goal * 0.5, start, end);
            let distance = (point - (ball_pos + ball_to_goal * 0.5)).norm();

            if distance < min_distance {
                min_distance = distance;
                closest_point = point;
            }
        }

        closest_point
    } else {
        Vector2::new(-1000.0, 0.0) // Fallback
    }
}

fn closest_point_on_segment(point: Vector2, start: Vector2, end: Vector2) -> Vector2 {
    let segment = end - start;
    let point_to_start = point - start;

    let t = (point_to_start.dot(&segment) / segment.dot(&segment))
        .max(0.0)
        .min(1.0);
    start + segment * t
}

fn evaluate_waller_positioning(s: &RobotSituation, ball_pos: Vector2, goal_pos: Vector2) -> f64 {
    let my_pos = s.player_data().position;

    // Check if we're between ball and goal
    let ball_to_goal = goal_pos - ball_pos;
    let ball_to_me = my_pos - ball_pos;

    // Project position onto ball-goal line
    let projection =
        (ball_to_me.x * ball_to_goal.x + ball_to_me.y * ball_to_goal.y) / ball_to_goal.norm();
    let projection_ratio = projection / ball_to_goal.norm();

    // Best position is about 30-70% between ball and goal
    if projection_ratio > 0.3 && projection_ratio < 0.7 {
        // Check lateral deviation
        let on_line_pos = ball_pos + ball_to_goal.normalize() * projection;
        let deviation = (my_pos - on_line_pos).norm();

        (1.0 - deviation / 1000.0).max(0.0)
    } else {
        0.0
    }
}
