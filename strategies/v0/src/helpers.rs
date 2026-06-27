//! Local, simplified stand-ins for the executor-internal helpers the v0 roles
//! used to call (`PassingStore`, `best_goal_shoot`, `find_best_receiver_target`,
//! `goal_shoot_success_probability`, the along-direction nearest-player scans).
//!
//! The originals lived inside the executor and had access to the full world +
//! tuned models. Out-of-process we re-derive equivalents from the
//! [`RobotSituation`] with plain geometry — good enough to reproduce the v0
//! behaviour for benchmarking, without pulling executor internals across the IPC
//! boundary.

use dies_core::{Vector2, PLAYER_RADIUS};
use dies_strategy_protocol::PlayerState;

use crate::bt::RobotSituation;

/// How clear the straight lane from `from` to `target` is of `obstacles`:
/// `1.0` when nothing blocks it, decaying to `0.0` as an obstacle sits on the
/// line. Only obstacles whose projection falls between the endpoints count.
fn lane_clearness(from: Vector2, target: Vector2, obstacles: &[PlayerState]) -> f64 {
    let seg = target - from;
    let len = seg.norm();
    if len < 1e-6 {
        return 1.0;
    }
    let dir = seg / len;
    let mut min_clear = 1.0_f64;
    for o in obstacles {
        let rel = o.position - from;
        let along = rel.dot(&dir);
        if along <= 0.0 || along >= len {
            continue;
        }
        let perp = (rel - dir * along).norm();
        // A robot within ~2 radii of the lane fully blocks it.
        let block_radius = PLAYER_RADIUS * 4.0;
        let clear = (perp / block_radius).clamp(0.0, 1.0);
        min_clear = min_clear.min(clear);
    }
    min_clear
}

/// Probability that a shot from `from` toward `goal` finds the net: lane
/// clearness against the opponents, scaled down with distance.
pub fn goal_shoot_success_probability(s: &RobotSituation, from: Vector2, goal: Vector2) -> f64 {
    let clear = lane_clearness(from, goal, &s.world.opp_players);
    let dist = (goal - from).norm();
    let dist_factor = (1.0 - (dist / 9000.0)).clamp(0.2, 1.0);
    clear * dist_factor
}

/// Best point in the opponent goal mouth to shoot at, and the success
/// probability of that shot from the ball's current position.
pub fn best_goal_shoot(s: &RobotSituation) -> (Vector2, f64) {
    let from = s.ball_position();
    let goal = s.get_opp_goal_position();
    let half_goal = s.field().goal_width / 2.0 - 60.0;

    // Sample aim points across the goal mouth, keep the clearest.
    let mut best_target = goal;
    let mut best_prob = f64::NEG_INFINITY;
    for i in 0..7 {
        let t = i as f64 / 6.0; // 0..1
        let y = -half_goal + t * (2.0 * half_goal);
        let target = Vector2::new(goal.x, y);
        let prob = goal_shoot_success_probability(s, from, target);
        if prob > best_prob {
            best_prob = prob;
            best_target = target;
        }
    }
    (best_target, best_prob.max(0.0))
}

/// Whether the ball-carrier has a reasonable shot on goal.
pub fn has_clear_shot(s: &RobotSituation) -> bool {
    best_goal_shoot(s).1 > 0.3
}

/// Pick an open forward position to receive/advance toward, biased toward the
/// opponent goal and away from opponents. `last` provides hysteresis so the
/// target doesn't jitter frame to frame.
pub fn find_best_receiver_target(s: &RobotSituation, last: Option<Vector2>) -> (Vector2, f64) {
    let goal = s.get_opp_goal_position();
    let half_w = s.half_field_width() - 400.0;

    // Candidate band ahead of the ball, on the attacking half.
    let base_x = s.ball_position().x.max(300.0);
    let mut best = Vector2::new(base_x + 1500.0, 0.0);
    let mut best_score = f64::NEG_INFINITY;

    for xi in 0..4 {
        for yi in -3..=3 {
            let x = (base_x + 600.0 * xi as f64).min(s.half_field_length() - 800.0);
            let y = (yi as f64 / 3.0) * half_w;
            let cand = Vector2::new(x, y);

            // Openness: distance to nearest opponent.
            let nearest_opp = s
                .world
                .opp_players
                .iter()
                .map(|p| (p.position - cand).norm())
                .fold(f64::INFINITY, f64::min);
            let open = (nearest_opp / 1500.0).clamp(0.0, 1.0);

            // Goal angle: closer/clearer to goal is better.
            let lane = lane_clearness(cand, goal, &s.world.opp_players);

            // Forward progress.
            let progress = (x / s.half_field_length()).clamp(0.0, 1.0);

            // Hysteresis: prefer staying near the previous target.
            let stay = match last {
                Some(p) => 1.0 - ((cand - p).norm() / 3000.0).clamp(0.0, 1.0),
                None => 0.0,
            };

            let score = open * 1.5 + lane * 1.5 + progress + stay * 0.5;
            if score > best_score {
                best_score = score;
                best = cand;
            }
        }
    }

    (s.constrain_to_field(best), best_score)
}

/// Distance to the nearest player (either team) found by walking a ray from
/// `from` along `dir`; `INFINITY` if the ray clears everyone. Mirrors the old
/// `find_nearest_player_distance_along_direction`.
pub fn nearest_player_distance_along_direction(
    s: &RobotSituation,
    from: Vector2,
    dir: Vector2,
) -> f64 {
    let dir = match dir.try_normalize(1e-6) {
        Some(d) => d,
        None => return f64::INFINITY,
    };
    let mut best = f64::INFINITY;
    let consider = s.world.own_players.iter().chain(s.world.opp_players.iter());
    for p in consider {
        if p.id == s.player_id {
            continue;
        }
        let rel = p.position - from;
        let along = rel.dot(&dir);
        if along <= 0.0 {
            continue;
        }
        let perp = (rel - dir * along).norm();
        if perp < PLAYER_RADIUS * 3.0 {
            best = best.min(along);
        }
    }
    best
}

/// As [`nearest_player_distance_along_direction`] but opponents only.
pub fn nearest_opponent_distance_along_direction(
    s: &RobotSituation,
    from: Vector2,
    dir: Vector2,
) -> f64 {
    let dir = match dir.try_normalize(1e-6) {
        Some(d) => d,
        None => return f64::INFINITY,
    };
    let mut best = f64::INFINITY;
    for p in &s.world.opp_players {
        let rel = p.position - from;
        let along = rel.dot(&dir);
        if along <= 0.0 {
            continue;
        }
        let perp = (rel - dir * along).norm();
        if perp < PLAYER_RADIUS * 3.0 {
            best = best.min(along);
        }
    }
    best
}
