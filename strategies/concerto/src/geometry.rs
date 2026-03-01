use dies_strategy_api::prelude::*;

/// Check whether the corridor from `ball_pos` to `goal_center` is free of opponents.
///
/// The corridor is a rectangle of the given `corridor_width` (perpendicular to the
/// ball→goal line). An opponent blocks the shot when its projection onto the segment
/// falls in [0, 1] **and** its perpendicular distance is less than `corridor_width / 2`.
pub fn is_clear_shot(
    ball_pos: Vector2,
    goal_center: Vector2,
    opponents: &[PlayerState],
    corridor_width: f64,
) -> bool {
    let dir = goal_center - ball_pos;
    let len_sq = dir.x * dir.x + dir.y * dir.y;

    // Degenerate case: ball is on top of the goal.
    if len_sq < 1e-6 {
        return true;
    }

    let half_w = corridor_width / 2.0;

    for opp in opponents {
        let to_opp = opp.position - ball_pos;

        // Projection parameter along the segment.
        let t = (to_opp.x * dir.x + to_opp.y * dir.y) / len_sq;

        if !(0.0..=1.0).contains(&t) {
            continue;
        }

        // Perpendicular distance from the line.
        let perp_dist = (to_opp.x * dir.y - to_opp.y * dir.x).abs() / len_sq.sqrt();

        if perp_dist < half_w {
            return false;
        }
    }

    true
}

/// Find the best area in the opponent half to pass the ball to.
///
/// Samples a grid of candidate positions in the attacking half (x > 0) and scores
/// each one by how far it is from the nearest opponent plus a bonus for having a
/// clear passing lane from `ball_pos`.
///
/// Returns `None` if every candidate scores below the minimum threshold (500 mm).
pub fn best_pass_area(
    ball_pos: Vector2,
    opponents: &[PlayerState],
    field_half_length: f64,
    field_half_width: f64,
) -> Option<Vector2> {
    let xs = [field_half_length * 0.25, field_half_length * 0.6];
    let ys = [
        -field_half_width * 0.4,
        -field_half_width * 0.15,
        0.0,
        field_half_width * 0.15,
        field_half_width * 0.4,
    ];

    const MIN_SCORE: f64 = 500.0;
    const CLEAR_LANE_BONUS: f64 = 300.0;
    const LANE_CORRIDOR: f64 = 200.0;

    let mut best: Option<(Vector2, f64)> = None;

    for &x in &xs {
        for &y in &ys {
            let candidate = Vector2::new(x, y);

            // Minimum distance to any opponent.
            let min_opp_dist = opponents
                .iter()
                .map(|o| (o.position - candidate).norm())
                .fold(f64::INFINITY, f64::min);

            // Clear-lane bonus: simplified corridor check between ball and candidate.
            let lane_bonus = if is_clear_shot(ball_pos, candidate, opponents, LANE_CORRIDOR) {
                CLEAR_LANE_BONUS
            } else {
                0.0
            };

            let score = min_opp_dist + 0.5 * lane_bonus;

            if score > MIN_SCORE
                && (best.is_none() || score > best.unwrap().1) {
                    best = Some((candidate, score));
                }
        }
    }

    best.map(|(pos, _)| pos)
}

/// Returns `true` if `point` lies roughly between `from` and `to` along the from→to axis.
///
/// Projects `point` onto the infinite line through `from` and `to` and checks whether
/// the parameter *t* falls in [0, 1].
pub fn is_between(point: Vector2, from: Vector2, to: Vector2) -> bool {
    let dir = to - from;
    let len_sq = dir.x * dir.x + dir.y * dir.y;

    if len_sq < 1e-6 {
        return false;
    }

    let to_point = point - from;
    let t = (to_point.x * dir.x + to_point.y * dir.y) / len_sq;

    (0.0..=1.0).contains(&t)
}

/// Compute two defensive "shadow" positions between the ball and our own goal.
///
/// 1. `direction` = normalize(own_goal − ball_pos).
/// 2. Place a centre point at `ball_pos + direction * offset_distance`.
/// 3. Perpendicular = direction rotated 90°.
/// 4. Return `(centre + perp * spread, centre − perp * spread)`.
///
/// `offset_distance` is clamped so the shadow positions don't end up behind the goal line.
pub fn compute_shadow_positions(
    ball_pos: Vector2,
    own_goal: Vector2,
    offset_distance: f64,
    spread: f64,
) -> (Vector2, Vector2) {
    let diff = own_goal - ball_pos;
    let dist_to_goal = diff.norm();

    // Avoid placing shadows behind the goal.
    let clamped_offset = offset_distance.min(dist_to_goal * 0.8);

    let direction = if dist_to_goal > 1e-6 {
        Vector2::new(diff.x / dist_to_goal, diff.y / dist_to_goal)
    } else {
        // Fallback: point toward −x (our goal side).
        Vector2::new(-1.0, 0.0)
    };

    let center = ball_pos + direction * clamped_offset;

    // Rotate direction 90° counter-clockwise for the perpendicular.
    let perp = Vector2::new(-direction.y, direction.x);

    (center + perp * spread, center - perp * spread)
}
