use dies_core::get_tangent_line_direction;
use dies_core::GameState;
use dies_core::Handicap;
use dies_core::{Angle, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn build_waller_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            committing_guard_node()
                .description("Pickup ball?")
                .when(should_pickup_ball)
                .until(should_not_pickup_ball)
                .commit_to(
                    semaphore_node()
                        .description("Pickup ball semaphore")
                        .semaphore_id("defender_pickup_ball".into())
                        .max_entry(1)
                        .do_then(
                            fetch_ball_with_preshoot()
                                .description("Pickup ball".to_string())
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .add(
            // Normal wall positioning
            continuous("wall position")
                .position(Argument::callback(|s| {
                    calculate_wall_position(s, "waller", false)
                }))
                .heading(get_defender_heading)
                .build(),
        )
        .description("Waller")
        .build()
        .into()
}

pub fn calculate_wall_position(s: &RobotSituation, role_name: &str, reverse: bool) -> Vector2 {
    // Generate all position tuples
    let position_tuples = generate_boundary_position_tuples(s, role_name);
    let position_tuples = if reverse {
        position_tuples
            .into_iter()
            .map(|t| t.into_iter().rev().collect())
            .collect()
    } else {
        position_tuples
    };

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
        .filter(|(_, v)| v.contains(role_name))
        .collect::<Vec<_>>();
    wallers.sort_by_key(|(k, _)| *k);

    let waller_index = wallers
        .iter()
        .position(|(k, _)| **k == s.player_id)
        .unwrap_or(0);

    let pos = if waller_index < best_tuple.len() {
        best_tuple[waller_index]
    } else {
        s.player_data().position
    };

    pos
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
    fn get_extended_boundary_segments(&self) -> Vec<(Vector2, Vector2)> {
        vec![
            // Top horizontal line
            (
                Vector2::new(self.goal_line_x - 10_000.0, self.top_y + 80.0), // extend beyond the field
                Vector2::new(self.penalty_line_x + 80.0, self.top_y + 50.0),
            ),
            // Right vertical line (penalty line)
            (
                Vector2::new(self.penalty_line_x + 80.0, self.top_y + 80.0),
                Vector2::new(self.penalty_line_x + 80.0, self.bottom_y - 80.0),
            ),
            // Bottom horizontal line
            (
                Vector2::new(self.penalty_line_x + 80.0, self.bottom_y - 80.0),
                Vector2::new(self.goal_line_x - 10_000.0, self.bottom_y - 80.0), // extend beyond the field
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

/// Generate candidate position tuples for all wallers
/// Returns array of position tuples, each containing (x,y) coordinates for each waller
pub fn generate_boundary_position_tuples(s: &RobotSituation, role_name: &str) -> Vec<Vec<Vector2>> {
    let mut wallers = s
        .role_assignments
        .iter()
        .filter(|(_, v)| v.contains(role_name))
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

    let segments = boundary.get_extended_boundary_segments();

    // Generate positions along boundary segments with 10mm spacing
    let mut boundary_positions = Vec::new();

    for (start, end) in segments {
        let segment_length = (end - start).norm();
        let num_positions = (segment_length / 10.0).ceil() as usize + 1;

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
    let max_tuples = 100;

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

                if distance >= 180.0 && distance <= 195.0 {
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

    if let Some(field) = &s.world.field_geom {
        let pos = position_tuple
            .iter()
            .fold(Vector2::new(0.0, 0.0), |acc, &v| acc + v)
            / position_tuple.len() as f64;
        let margin = 150.0; // at what distance the ball in penalty area is safe from attackers
        let ball_pos = ball.position.xy();
        let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
        let y_bound = field.penalty_area_width / 2.0 - margin;
        if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
            // in this case we actually want to just stay somewhere non-blocking
            return (ball_pos - pos).norm() - (pos - s.player_data().position).norm() * 0.5;
            // bigger is better thus
        }
    }

    let ball_pos = ball.position.xy();
    let goal_pos = s.get_own_goal_position();
    let mut total_score = 0.0;

    let Some(boundary) = get_defense_area_boundary(s) else {
        return 0.0;
    };

    let backline = boundary.get_back_line();
    for &pos in position_tuple.iter().take(1) {
        // the closer to ball - the better
        let ball_score = -(ball_pos - pos).norm() / 5_000.0;
        total_score += ball_score;

        // how much of the backline is covered by the robot at this position
        let coverage_score = compute_coverage_score(ball_pos, pos, backline);
        total_score += coverage_score * 0.0; // Weight coverage highly

        // the closer the position to the ball-goal ray the better
        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_pos = pos - ball_pos;
        let projection = ball_to_pos.dot(&ball_to_goal.normalize());
        let perpendicular_dist = (ball_to_pos - ball_to_goal.normalize() * projection).norm();

        let ray_score = -perpendicular_dist / 1000.0; // Penalize positions far from ball-goal line
        total_score += ray_score;
    }

    // Penalize positions too far from current robot positions
    let mut wallers = s
        .role_assignments
        .iter()
        .filter(|(_, v)| v.contains("waller"))
        .collect::<Vec<_>>();
    wallers.sort_by_key(|(k, _)| *k);

    total_score
}

fn should_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return false;
    }
    if s.has_any_handicap(&[Handicap::NoKicker, Handicap::NoDribbler]) {
        return false;
    }
    if !s.can_touch_ball() {
        return false;
    }

    let no_harassers = s
        .role_assignments
        .values()
        .find(|v| v.contains("harasser"))
        .is_none();

    let ball_in_our_half = s.ball_position().x < 0.0;
    let ball_is_slow = s.ball_speed() < 200.0;
    let opponents_are_far = s.distance_of_closest_opp_player_to_ball() > 1000.0;
    let i_am_the_closest_robot = {
        let my_distance = s.distance_to_ball();
        let ball_pos = s.ball_position();

        // Check if I'm closer than all teammates
        s.world.own_players.iter().all(|robot_data| {
            let teammate_distance = (robot_data.position.xy() - ball_pos).norm();
            my_distance <= teammate_distance
        })
    };

    let should = no_harassers
        && ball_in_our_half
        && !s.ball_in_own_penalty_area()
        && ball_is_slow
        && opponents_are_far
        && i_am_the_closest_robot;

    should
}

fn should_not_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return true;
    }

    s.distance_of_closest_opp_player_to_ball() < 800.0
        || s.distance_of_closest_own_player_to_ball() < 200.0
}

pub fn get_defender_heading(s: &RobotSituation) -> Angle {
    if let Some(ball) = &s.world.ball {
        let ball_pos = ball.position.xy();
        let my_pos = s.player_data().position;
        Angle::between_points(my_pos, ball_pos)
    } else {
        Angle::from_radians(0.0)
    }
}
