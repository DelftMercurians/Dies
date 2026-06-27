use dies_core::{get_tangent_line_direction, Angle, Vector2, PLAYER_RADIUS};
use dies_strategy_protocol::{GameState, Handicap};

use crate::bt::*;

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
                        .semaphore_id("defender_pickup_ball")
                        .max_entry(1)
                        .do_then(
                            fetch_ball_with_preshoot()
                                .description("Pickup ball")
                                .build(),
                        )
                        .build(),
                )
                .build(),
        )
        .add(
            continuous("wall position")
                .position(Argument::callback(|s| {
                    calculate_wall_position(s, "waller", false)
                }))
                .heading(get_defender_heading as fn(&RobotSituation) -> Angle)
                .build(),
        )
        .description("Waller")
        .build()
        .into()
}

pub fn calculate_wall_position(s: &RobotSituation, role_name: &str, reverse: bool) -> Vector2 {
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
        return s.position();
    }

    let mut best_tuple = &position_tuples[0];
    let mut best_score = score_position_tuple(s, best_tuple);
    for tuple in &position_tuples[1..] {
        let score = score_position_tuple(s, tuple);
        if score > best_score {
            best_score = score;
            best_tuple = tuple;
        }
    }

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

    if waller_index < best_tuple.len() {
        best_tuple[waller_index]
    } else {
        s.position()
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
            (
                Vector2::new(self.goal_line_x - 10_000.0, self.top_y + 80.0),
                Vector2::new(self.penalty_line_x + 80.0, self.top_y + 50.0),
            ),
            (
                Vector2::new(self.penalty_line_x + 80.0, self.top_y + 80.0),
                Vector2::new(self.penalty_line_x + 80.0, self.bottom_y - 80.0),
            ),
            (
                Vector2::new(self.penalty_line_x + 80.0, self.bottom_y - 80.0),
                Vector2::new(self.goal_line_x - 10_000.0, self.bottom_y - 80.0),
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

fn get_defense_area_boundary(s: &RobotSituation) -> Option<DefenseAreaBoundary> {
    s.world.field_geom.as_ref().map(|field| {
        let goal_line_x = -field.field_length / 2.0;
        let penalty_line_x = goal_line_x + field.penalty_area_depth;
        let half_width = field.penalty_area_width / 2.0;
        DefenseAreaBoundary {
            goal_line_x,
            penalty_line_x,
            top_y: half_width,
            bottom_y: -half_width,
        }
    })
}

fn line_intersection(p1: Vector2, p2: Vector2, p3: Vector2, p4: Vector2) -> Option<Vector2> {
    let denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if denom.abs() < 1e-10 {
        return None;
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
        return false;
    }
    let dot_product =
        (point.x - start.x) * (end.x - start.x) + (point.y - start.y) * (end.y - start.y);
    let squared_length = (end.x - start.x).powi(2) + (end.y - start.y).powi(2);
    dot_product >= 0.0 && dot_product <= squared_length
}

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
    if s.world.ball.is_none() {
        return vec![];
    }

    let segments = boundary.get_extended_boundary_segments();

    let mut boundary_positions = Vec::new();
    for (start, end) in segments {
        let segment_length = (end - start).norm();
        let num_positions = (segment_length / 10.0).ceil() as usize + 1;
        for i in 0..num_positions {
            let t = i as f64 / (num_positions - 1) as f64;
            let pos = start + (end - start) * t;
            if boundary_positions.is_empty()
                || pos != boundary_positions[boundary_positions.len() - 1]
            {
                boundary_positions.push(pos);
            }
        }
    }

    let mut position_tuples = Vec::new();
    let max_tuples = 100;
    let step = (boundary_positions.len() / max_tuples).max(1);

    for i in (0..boundary_positions.len()).step_by(step) {
        if position_tuples.len() >= max_tuples {
            break;
        }

        let mut tuple = Vec::new();
        let mut current_idx = i;
        tuple.push(boundary_positions[current_idx]);

        for _ in 1..waller_count {
            let current_pos = boundary_positions[current_idx];
            let mut next_idx = current_idx + 1;
            while next_idx < boundary_positions.len() {
                let next_pos = boundary_positions[next_idx];
                let distance = (next_pos - current_pos).norm();
                if (180.0..=195.0).contains(&distance) {
                    tuple.push(next_pos);
                    current_idx = next_idx;
                    break;
                }
                next_idx += 1;
            }
        }

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
    let (angle1, angle2) = get_tangent_line_direction(our_pos, PLAYER_RADIUS, ball);

    let dir1 = Vector2::new(angle1.radians().cos(), angle1.radians().sin());
    let dir2 = Vector2::new(angle2.radians().cos(), angle2.radians().sin());

    let far_point1 = ball + dir1 * 10000.0;
    let far_point2 = ball + dir2 * 10000.0;

    let intersection1 = line_intersection(ball, far_point1, backline.0, backline.1);
    let intersection2 = line_intersection(ball, far_point2, backline.0, backline.1);

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

    0.0
}

pub fn score_position_tuple(s: &RobotSituation, position_tuple: &[Vector2]) -> f64 {
    let Some(ball) = &s.world.ball else {
        return 0.0;
    };
    let ball_pos = ball.position;
    let goal_pos = s.get_own_goal_position();
    let mut total_score = 0.0;

    let Some(boundary) = get_defense_area_boundary(s) else {
        return 0.0;
    };
    let backline = boundary.get_back_line();

    for &pos in position_tuple.iter().take(1) {
        let ball_score = -(ball_pos - pos).norm() / 5_000.0;
        total_score += ball_score;

        let coverage_score = compute_coverage_score(ball_pos, pos, backline);
        total_score += coverage_score * 0.0;

        let ball_to_goal = goal_pos - ball_pos;
        let ball_to_pos = pos - ball_pos;
        let projection = ball_to_pos.dot(&ball_to_goal.normalize());
        let perpendicular_dist = (ball_to_pos - ball_to_goal.normalize() * projection).norm();

        let ray_score = -perpendicular_dist / 1000.0;
        total_score += ray_score;
    }

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

    let no_harassers = !s.role_assignments.values().any(|v| v.contains("harasser"));
    let ball_in_our_half = s.ball_position().x < 0.0;
    let ball_is_slow = s.ball_speed() < 200.0;
    let opponents_are_far = s.distance_of_closest_opp_player_to_ball() > 1000.0;
    let i_am_the_closest_robot = {
        let my_distance = s.distance_to_ball();
        let ball_pos = s.ball_position();
        s.world
            .own_players
            .iter()
            .all(|robot| (robot.position - ball_pos).norm() >= my_distance)
    };

    no_harassers
        && ball_in_our_half
        && !s.ball_in_own_penalty_area()
        && ball_is_slow
        && opponents_are_far
        && i_am_the_closest_robot
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
        Angle::between_points(s.position(), ball.position)
    } else {
        Angle::from_radians(0.0)
    }
}
