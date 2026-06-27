use dies_strategy_api::prelude::*;

/// Momentum-aware estimate of the time (seconds) for a robot at `pos` moving with
/// `vel` to reach `target`, given top speed `v_max` and acceleration `a_max`.
///
/// Unlike Euclidean distance, this credits velocity already pointing at the target
/// and penalises velocity that must be reversed or bled off sideways. A robot
/// driving toward a target has genuinely low cost to continue and high cost to
/// turn around — the physical basis for assignment/selection stability (no
/// artificial stay-bonus).
pub fn redirect_time(pos: Vector2, vel: Vector2, target: Vector2, v_max: f64, a_max: f64) -> f64 {
    let d = target - pos;
    let dist = d.norm();
    if dist < 1e-3 {
        return 0.0;
    }
    let dir = d / dist;

    let v_along = vel.dot(&dir); // + toward target, - away
    let t_cruise = dist / v_max;

    let v_against = (-v_along).max(0.0); // wrong-way speed to reverse
    let v_cross = (vel - dir * v_along).norm(); // sideways speed to bleed off
    let t_redirect = (2.0 * v_against + v_cross) / a_max;

    // Small head-start credit for velocity already toward the target.
    let t_credit = v_along.max(0.0).min(v_max) / (2.0 * a_max);

    (t_cruise + t_redirect - t_credit).max(0.0)
}

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
///
/// Reserved for the passing milestone (planner's no-clear-shot branch).
#[allow(dead_code)]
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

            if score > MIN_SCORE && (best.is_none() || score > best.unwrap().1) {
                best = Some((candidate, score));
            }
        }
    }

    best.map(|(pos, _)| pos)
}

/// Pick an open, forward support/outlet position on one flank.
///
/// Static flank points leave supporters stranded behind opponents, so the ball
/// can never reach them. This samples a small grid *ahead of the ball* on the
/// given flank (`sign` = ±1 for the +y / -y side) and returns the candidate with
/// the most open ball→candidate lane, mildly preferring more forward positions.
/// The result is always a valid outlet the carrier could actually pass into.
pub fn best_support_pos(
    ball: Vector2,
    opponents: &[PlayerState],
    sign: f64,
    half_len: f64,
    half_wid: f64,
    corridor: f64,
) -> Vector2 {
    const PAD: f64 = 400.0;
    let x_offsets = [1000.0, 2000.0, 3000.0];
    let y_fracs = [0.18, 0.32, 0.46];

    let mut best: Option<(Vector2, f64)> = None;
    for &dx in &x_offsets {
        let cx = (ball.x + dx).clamp(-half_len + PAD, half_len - PAD);
        for &fy in &y_fracs {
            let cy = (sign * half_wid * fy).clamp(-half_wid + PAD, half_wid - PAD);
            let cand = Vector2::new(cx, cy);
            // Openness dominates (keeps the supporter out from behind an opponent);
            // the small forward bonus only breaks ties between open candidates.
            let score = lane_openness(ball, cand, opponents, corridor) + 5e-5 * cx;
            if best.is_none() || score > best.unwrap().1 {
                best = Some((cand, score));
            }
        }
    }
    best.map(|(p, _)| p).unwrap_or_else(|| {
        Vector2::new(
            (ball.x + 2000.0).clamp(-half_len + PAD, half_len - PAD),
            sign * half_wid * 0.3,
        )
    })
}

/// Returns `true` if `point` lies roughly between `from` and `to` along the from→to axis.
///
/// Projects `point` onto the infinite line through `from` and `to` and checks whether
/// the parameter *t* falls in [0, 1].
///
/// Reserved for the M3 conservative steal gate (is a defender between threat and goal?).
#[allow(dead_code)]
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

/// Direction to break out of a ball contest: a unit vector **perpendicular to the
/// carrier→presser squeeze axis**, on whichever side is more open.
///
/// In a pin both dribblers push along the squeeze axis, so the perpendicular is
/// the one unopposed escape direction. Of the two perpendiculars we pick the side
/// with the more open lane from the ball (via [`lane_openness`]), mildly biased
/// toward midfield so we don't shepherd the ball into a touchline. Robust to the
/// degenerate case where carrier and presser coincide.
pub fn escape_direction(
    carrier: Vector2,
    presser: Vector2,
    ball: Vector2,
    opponents: &[PlayerState],
    field_half_width: f64,
) -> Vector2 {
    const PROBE_DIST: f64 = 1000.0;
    const CORRIDOR: f64 = 400.0;
    const CENTER_BIAS: f64 = 0.25;

    // Squeeze axis (carrier → presser), with finite fallbacks if robots coincide.
    let mut axis = presser - carrier;
    if axis.norm() < 1e-6 {
        axis = ball - carrier;
    }
    if axis.norm() < 1e-6 {
        axis = Vector2::new(1.0, 0.0);
    }
    let axis = axis / axis.norm();
    let perp = Vector2::new(-axis.y, axis.x); // unit ⟂ to the axis

    let half_w = field_half_width.max(1.0);
    let score = |dir: Vector2| -> f64 {
        let cand = ball + dir * PROBE_DIST;
        let openness = lane_openness(ball, cand, opponents, CORRIDOR);
        let center = 1.0 - (cand.y.abs() / half_w).min(1.0);
        openness + CENTER_BIAS * center
    };

    if score(perp) >= score(-perp) {
        perp
    } else {
        -perp
    }
}

/// Smooth Hermite step: 0 below `edge0`, 1 above `edge1`, C¹-continuous between.
/// Used everywhere a hard threshold would otherwise cause formation discontinuities.
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if (edge1 - edge0).abs() < 1e-9 {
        return if x < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Continuous threat a field position poses to our goal, in [0, 1].
///
/// Combines proximity to our goal (closer = higher) with a directional term
/// (square-on to the goal mouth is worse than wide). Smooth in `pos`.
pub fn threat(pos: Vector2, own_goal: Vector2, goal_near: f64, goal_far: f64) -> f64 {
    let to_goal = own_goal - pos;
    let dist = to_goal.norm();
    let prox = 1.0 - smoothstep(goal_near, goal_far, dist);
    if dist < 1e-6 {
        return prox;
    }
    // Directional cosine toward our goal (pos on the -x side aiming at goal = 1).
    let aim = (to_goal / dist).dot(&Vector2::new(-1.0, 0.0)).max(0.0);
    prox * (0.5 + 0.5 * aim)
}

/// Continuous openness in [0, 1] of the lane from `from` to `to` w.r.t. opponents.
///
/// 1 if no opponent is near the segment, decaying toward 0 as the nearest in-corridor
/// opponent approaches the line. The smooth analogue of [`is_clear_shot`].
pub fn lane_openness(from: Vector2, to: Vector2, opponents: &[PlayerState], corridor: f64) -> f64 {
    let dir = to - from;
    let len = dir.norm();
    if len < 1e-6 {
        return 1.0;
    }
    let mut min_perp = f64::INFINITY;
    for opp in opponents {
        let to_opp = opp.position - from;
        let t = to_opp.dot(&dir) / (len * len);
        if !(0.0..=1.0).contains(&t) {
            continue;
        }
        let perp = (to_opp.x * dir.y - to_opp.y * dir.x).abs() / len;
        min_perp = min_perp.min(perp);
    }
    if min_perp.is_infinite() {
        return 1.0;
    }
    // 0 when on the line, 1 when at/over the corridor edge.
    smoothstep(0.0, corridor, min_perp)
}

/// Distribute `k` shadow positions across the goal-mouth coverage arc as seen from
/// `ball`, on a standoff line in front of our goal.
///
/// Positions vary continuously with the ball; ordered left→right so slot identity
/// is stable as the fan rotates. `standoff` is how far in front of the goal the
/// shadows sit (clamped so they stay in front of the ball).
pub fn shadow_arc(
    ball: Vector2,
    own_goal: Vector2,
    k: usize,
    standoff: f64,
    half_goal: f64,
) -> Vec<Vector2> {
    if k == 0 {
        return Vec::new();
    }
    let dist = (own_goal - ball).norm();
    let line_x = own_goal.x + standoff.min(dist * 0.8);

    // Aim points span the goal mouth; each shadow blocks the ball→aim ray at line_x.
    (0..k)
        .map(|i| {
            // Spread aim across the mouth, left→right. Single shadow → centre.
            let frac = if k == 1 {
                0.5
            } else {
                i as f64 / (k as f64 - 1.0)
            };
            let aim_y = -half_goal + 2.0 * half_goal * frac;
            let aim = Vector2::new(own_goal.x, aim_y);
            // Intersect ball→aim with x = line_x.
            let d = aim - ball;
            let y = if d.x.abs() < 1e-6 {
                ball.y
            } else {
                let t = (line_x - ball.x) / d.x;
                ball.y + t * d.y
            };
            Vector2::new(line_x, y)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redirect_time_prefers_continuing_over_reversing() {
        let pos = Vector2::new(0.0, 0.0);
        let target = Vector2::new(1000.0, 0.0);
        let toward = redirect_time(pos, Vector2::new(2000.0, 0.0), target, 3000.0, 3000.0);
        let away = redirect_time(pos, Vector2::new(-2000.0, 0.0), target, 3000.0, 3000.0);
        let still = redirect_time(pos, Vector2::new(0.0, 0.0), target, 3000.0, 3000.0);
        assert!(
            toward < still,
            "moving toward should be cheaper than stationary"
        );
        assert!(
            still < away,
            "stationary should be cheaper than moving away"
        );
    }

    #[test]
    fn smoothstep_is_monotonic_and_clamped() {
        assert_eq!(smoothstep(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 2.0), 1.0);
        let a = smoothstep(0.0, 1.0, 0.25);
        let b = smoothstep(0.0, 1.0, 0.75);
        assert!(a < b);
    }

    #[test]
    fn threat_is_higher_near_our_goal() {
        let own_goal = Vector2::new(-4500.0, 0.0);
        let near = threat(Vector2::new(-3500.0, 0.0), own_goal, 1500.0, 6000.0);
        let far = threat(Vector2::new(3000.0, 0.0), own_goal, 1500.0, 6000.0);
        assert!(near > far);
    }

    #[test]
    fn support_pos_avoids_standing_behind_opponent() {
        // An opponent sits straight ahead on the +y flank; the supporter must pick
        // an open lane instead of parking behind it.
        let ball = Vector2::new(0.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(2000.0, 960.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let pos = best_support_pos(
            ball,
            std::slice::from_ref(&blocker),
            1.0,
            4500.0,
            3000.0,
            500.0,
        );
        let openness = lane_openness(ball, pos, std::slice::from_ref(&blocker), 500.0);
        assert!(pos.y > 0.0, "support should stay on its flank");
        assert!(
            openness > 0.9,
            "support lane should be open, got {openness} at {pos:?}"
        );
    }

    #[test]
    fn escape_direction_is_lateral_and_avoids_the_blocked_side() {
        // Presser dead ahead along +x → escape must be lateral (±y). An opponent
        // blocks the +y lane, so we must squirt out toward -y.
        let carrier = Vector2::new(0.0, 0.0);
        let presser = Vector2::new(500.0, 0.0);
        let ball = Vector2::new(250.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(250.0, 1000.0), // sitting in the +y escape lane
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let dir = escape_direction(
            carrier,
            presser,
            ball,
            std::slice::from_ref(&blocker),
            3000.0,
        );
        assert!(
            dir.x.abs() < 1e-6,
            "escape should be perpendicular to the axis"
        );
        assert!(
            dir.y < 0.0,
            "escape should avoid the blocked +y side, got {dir:?}"
        );
        assert!(
            (dir.norm() - 1.0).abs() < 1e-6,
            "escape should be a unit vector"
        );
    }

    #[test]
    fn lane_openness_drops_when_blocked() {
        let from = Vector2::new(0.0, 0.0);
        let to = Vector2::new(2000.0, 0.0);
        let blocker = PlayerState::new(
            PlayerId::new(9),
            Vector2::new(1000.0, 0.0),
            Vector2::new(0.0, 0.0),
            Angle::from_radians(0.0),
        );
        let blocked = lane_openness(from, to, std::slice::from_ref(&blocker), 500.0);
        let clear = lane_openness(from, to, &[], 500.0);
        assert!(blocked < clear);
        assert_eq!(clear, 1.0);
    }
}
