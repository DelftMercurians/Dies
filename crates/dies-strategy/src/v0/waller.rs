use dies_core::get_tangent_line_direction;
use dies_core::Vector2;
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::{evaluate_ball_threat, get_defender_heading};

pub fn build_waller_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            // Normal wall positioning
            continuous("wall position")
                .position(calculate_wall_position)
                .heading(get_defender_heading)
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
    // Generate all position tuples
    let position_tuples = generate_boundary_position_tuples(s);

    if position_tuples.is_empty() {
        return s.player_data().position;
    }

    // Find the best tuple
    let mut best_tuple = &position_tuples[0];
    let mut best_score = score_position_tuple(s, best_tuple);

    for tuple in &position_tuples[1..] {
        let score = score_position_tuple(s, tuple);
        if score > best_score {
            best_score = score;
            best_tuple = tuple;
        }
    }

    // Find this waller's index and return corresponding position
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

    if waller_index < best_tuple.len() {
        best_tuple[waller_index]
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
        ]
    }

    fn get_back_line(&self) -> (Vector2, Vector2) {
        (
            Vector2::new(self.goal_line_x, self.top_y),
            Vector2::new(self.goal_line_x, self.bottom_y),
        )
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

/// Generate candidate position tuples for all wallers
/// Returns array of position tuples, each containing (x,y) coordinates for each waller
pub fn generate_boundary_position_tuples(s: &RobotSituation) -> Vec<Vec<Vector2>> {
    let mut wallers = s
        .role_assignments
        .iter()
        .filter(|(_, v)| *v == "waller")
        .collect::<Vec<_>>();
    wallers.sort_by_key(|(k, _)| *k);

    let waller_count = wallers.len();
    if waller_count == 0 {
        return vec![];
    }

    let Some(boundary) = get_defense_area_boundary(s) else {
        return vec![];
    };

    let Some(ball) = &s.world.ball else {
        return vec![];
    };

    let segments = boundary.get_boundary_segments();

    // Generate positions along boundary segments with 200mm spacing
    let mut boundary_positions = Vec::new();

    for (start, end) in segments {
        let segment_length = (end - start).norm();
        let num_positions = (segment_length / 100.0).ceil() as usize + 1;

        for i in 0..num_positions {
            let t = i as f64 / (num_positions - 1) as f64;
            let pos = start + (end - start) * t;
            if boundary_positions.len() == 0
                || pos != boundary_positions[boundary_positions.len() - 1]
            {
                boundary_positions.push(pos);
            }
        }
    }

    let mut position_tuples = Vec::new();
    let max_tuples = 50;

    // Generate 50 tuples by sampling leftmost waller positions
    let step = (boundary_positions.len() / max_tuples).max(1);

    for i in (0..boundary_positions.len()).step_by(step) {
        if position_tuples.len() >= max_tuples {
            break;
        }

        let mut tuple = Vec::new();
        let mut current_idx = i;

        // Place first waller at the sampled position
        tuple.push(boundary_positions[current_idx]);

        // Place subsequent wallers 200mm to the next available positions
        for _ in 1..waller_count {
            // Find next position that's at least 200mm away
            let current_pos = boundary_positions[current_idx];
            let mut next_idx = current_idx + 1;

            while next_idx < boundary_positions.len() {
                let next_pos = boundary_positions[next_idx];
                let distance = (next_pos - current_pos).norm();

                if distance >= 170.0 && distance <= 200.0 {
                    tuple.push(next_pos);
                    current_idx = next_idx;
                    break;
                }
                next_idx += 1;
            }
        }

        // Only add tuple if we successfully placed all wallers
        if tuple.len() == waller_count {
            position_tuples.push(tuple);
        }
    }

    position_tuples
}

pub fn compute_coverage_score(
    ball: Vector2,
    our_pos: Vector2,
    backline: (Vector2, Vector2),
) -> f64 {
    // Calculate the two tangent line directions from ball to robot circle
    let (angle1, angle2) = get_tangent_line_direction(our_pos, dies_core::PLAYER_RADIUS, ball);

    // Convert angles to direction vectors
    let dir1 = Vector2::new(angle1.radians().cos(), angle1.radians().sin());
    let dir2 = Vector2::new(angle2.radians().cos(), angle2.radians().sin());

    // Create far points for the shadow rays (extending the tangent lines)
    let far_point1 = ball + dir1 * 10000.0;
    let far_point2 = ball + dir2 * 10000.0;

    // Find intersections with the goal line (backline)
    let intersection1 = line_intersection(ball, far_point1, backline.0, backline.1);
    let intersection2 = line_intersection(ball, far_point2, backline.0, backline.1);

    // Check if both intersections exist and are on the goal line segment
    if let (Some(p1), Some(p2)) = (intersection1, intersection2) {
        let total_length = (backline.1 - backline.0).norm();
        let mut coverage_length = 0.0;

        if is_point_on_segment(p1, backline.0, backline.1)
            && is_point_on_segment(p2, backline.0, backline.1)
        {
            coverage_length = (p2 - p1).norm();
        } else if is_point_on_segment(p1, backline.0, backline.1) {
            coverage_length = (p1 - backline.0).norm().min((p1 - backline.1).norm());
        } else if is_point_on_segment(p2, backline.0, backline.1) {
            coverage_length = (p2 - backline.0).norm().min((p2 - backline.1).norm());
        }

        return coverage_length / total_length;
    }

    // If no valid coverage found, return 0
    0.0
}

/// Score a position tuple for all wallers
/// Higher score is better
pub fn score_position_tuple(s: &RobotSituation, position_tuple: &[Vector2]) -> f64 {
    let Some(ball) = &s.world.ball else {
        return 0.0;
    };

    let ball_pos = ball.position.xy();
    let goal_pos = s.get_own_goal_position();
    let mut total_score = 0.0;

    let Some(boundary) = get_defense_area_boundary(s) else {
        return 0.0;
    };

    let backline = boundary.get_back_line();

    for &pos in position_tuple {
        // the closer to ball - the better
        let ball_score = -(ball_pos - pos).norm() / 5_000.0;
        total_score += ball_score;

        // how much of the backline is covered by the robot at this position
        let coverage_score = compute_coverage_score(ball_pos, pos, backline);
        total_score += coverage_score * 100.0; // Weight coverage highly

        // Base scoring using existing evaluation
        let pos_score = evaluate_boundary_position(s, pos, ball_pos, goal_pos);
        total_score += pos_score;
    }

    // Penalize positions too far from current robot positions
    let mut wallers = s
        .role_assignments
        .iter()
        .filter(|(_, v)| *v == "waller")
        .collect::<Vec<_>>();
    wallers.sort_by_key(|(k, _)| *k);

    for (i, (player_id, _)) in wallers.iter().enumerate() {
        if i < position_tuple.len() {
            let player = s.world.get_player(**player_id);
            let current_pos = player.position;
            let target_pos = position_tuple[i];
            let distance = (current_pos - target_pos).norm();

            // Slight penalty for distance to encourage stability
            total_score -= distance * 0.01;
        }
    }

    total_score
}
