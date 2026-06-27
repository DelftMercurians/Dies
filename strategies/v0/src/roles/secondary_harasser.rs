use dies_core::{Angle, Vector2};
use dies_strategy_protocol::{GameState, PlayerId, PlayerState};

use crate::bt::*;

use super::harasser::{calculate_primary_harasser_position, should_harasser_cancel_pickup_ball};
use super::utils::fetch_and_shoot;

const TAGGING_DISTANCE: f64 = 300.0;
const HYSTERESIS_FACTOR: f64 = 1.3;

pub fn build_secondary_harasser_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            committing_guard_node()
                .when(should_harasser_pickup_ball)
                .until(should_harasser_cancel_pickup_ball)
                .commit_to(
                    semaphore_node()
                        .semaphore_id("defender_pickup_ball")
                        .max_entry(1)
                        .do_then(fetch_and_shoot())
                        .build(),
                )
                .build(),
        )
        .add(
            stateful_continuous("tagging_harasser")
                .with_stateful_position(move |s, last: Option<&Option<PlayerId>>| {
                    let last_id = last.copied().flatten();
                    let (target_player_id, pos) = calculate_secondary_harasser_position(s, last_id);
                    (pos, Some(target_player_id))
                })
                .with_stateful_heading(|s, _: Option<&Angle>| {
                    let heading = Angle::between_points(s.position(), s.ball_position());
                    (heading, Some(heading))
                })
                .build(),
        )
        .build()
        .into()
}

fn calculate_secondary_harasser_position(
    s: &RobotSituation,
    last_target_player_id: Option<PlayerId>,
) -> (Option<PlayerId>, Vector2) {
    let Some(ball) = &s.world.ball else {
        return (None, Vector2::new(-2000.0, -1000.0));
    };
    let ball_pos = ball.position;

    let target_opponent = find_best_opponent_to_tag(s, ball_pos, last_target_player_id);

    let Some(opponent) = target_opponent else {
        return (None, calculate_primary_harasser_position(s));
    };

    // Position between the opponent and the ball.
    let opp_pos = opponent.position;
    let ball_to_opp = (opp_pos - ball_pos).normalize();
    let tag_position = opp_pos - ball_to_opp * TAGGING_DISTANCE;

    (
        Some(opponent.id),
        Vector2::new(tag_position.x.min(-100.0), tag_position.y),
    )
}

/// Threat of an opponent: how close they are to our goal, with a bonus for being
/// unmarked. (Originally derived from a shoot-probability model in the executor;
/// re-derived here from plain geometry.)
fn opponent_threat(s: &RobotSituation, opponent: &PlayerState) -> f64 {
    let own_goal = s.get_own_goal_position();
    let dist_to_goal = (opponent.position - own_goal).norm();
    let base = 1.0 / (1.0 + dist_to_goal / 1000.0);

    let closest_own_harasser = s
        .get_players_with_role("harasser")
        .into_iter()
        .filter(|p| p.id != s.player_id())
        .min_by_key(|p| (p.position - opponent.position).norm() as i64);
    let unmarked = match closest_own_harasser {
        Some(h) if (opponent.position - h.position).norm() < 400.0 => 0.3,
        _ => 0.0,
    };

    base + unmarked
}

fn find_best_opponent_to_tag(
    s: &RobotSituation,
    ball_pos: Vector2,
    current_target_id: Option<PlayerId>,
) -> Option<PlayerState> {
    let mut best_opponent = None;
    let mut best_score = 0.0;
    let mut current_target_score = 0.0;

    for opponent in &s.world.opp_players {
        // Only opponents on our half, not the one with the ball.
        if opponent.position.x >= 0.0 {
            continue;
        }
        if (opponent.position - ball_pos).norm() < 300.0 {
            continue;
        }

        let final_score = opponent_threat(s, opponent);

        if Some(opponent.id) == current_target_id {
            current_target_score = final_score;
        }
        if final_score > best_score {
            best_score = final_score;
            best_opponent = Some(opponent);
        }
    }

    // Hysteresis: keep the current target unless a new one is clearly better.
    if let Some(current_id) = current_target_id {
        if current_target_score > 0.0 && best_score < current_target_score * HYSTERESIS_FACTOR {
            if let Some(current_opp) = s.world.opp_players.iter().find(|p| p.id == current_id) {
                return Some(current_opp.clone());
            }
        }
    }

    best_opponent.cloned()
}

fn should_harasser_pickup_ball(s: &RobotSituation) -> bool {
    if s.game_state_is_not(GameState::Run) {
        return false;
    }
    if !s.can_touch_ball() {
        return false;
    }

    if let (Some(field), Some(ball)) = (&s.world.field_geom, &s.world.ball) {
        let ball_pos = ball.position;

        let margin = 150.0;
        let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
        let y_bound = field.penalty_area_width / 2.0 - margin;
        if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
            return false;
        }

        if ball_pos.x >= 70.0 {
            return false;
        }

        if ball.velocity.norm() > 1000.0 {
            return false;
        }

        s.am_closest_to_ball()
    } else {
        false
    }
}
