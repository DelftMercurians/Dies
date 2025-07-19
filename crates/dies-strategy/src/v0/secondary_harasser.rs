use dies_core::{distance_to_line, Angle, GameState, PlayerData, PlayerId, Vector2};
use dies_executor::{behavior_tree_api::*, goal_shoot_success_probability, PassingStore};

use crate::v0::{
    harasser::{calculate_primary_harasser_position, should_harasser_cancel_pickup_ball},
    utils::fetch_and_shoot,
};

const TAGGING_DISTANCE: f64 = 300.0; // Distance to maintain from tagged opponent
const HYSTERESIS_FACTOR: f64 = 1.3; // Current target needs to be 30% worse before switching

pub fn build_secondary_harasser_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            committing_guard_node()
                .when(|s| should_harasser_pickup_ball(s))
                .until(should_harasser_cancel_pickup_ball)
                .commit_to(
                    semaphore_node()
                        .semaphore_id("defender_pickup_ball".to_string())
                        .max_entry(1)
                        .do_then(fetch_and_shoot())
                        .build(),
                )
                .build(),
        )
        .add(
            stateful_continuous("tagging_harasser")
                .with_stateful_position(move |s, last_target_player_id| {
                    let (target_player_id, pos) =
                        calculate_secondary_harasser_position(s, last_target_player_id.copied());
                    (pos, target_player_id)
                })
                .with_stateful_heading(|s, _| {
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

    let ball_pos = ball.position.xy();
    let passing_store = PassingStore::from(s);

    // Find the best opponent to tag (on our side of field, not currently with ball)
    let target_opponent =
        find_best_opponent_to_tag(s, &passing_store, ball_pos, last_target_player_id);

    let Some(opponent) = target_opponent else {
        return (None, calculate_primary_harasser_position(s));
    };

    // Position between the opponent and the ball
    let opp_pos = opponent.position;
    let ball_to_opp = (opp_pos - ball_pos).normalize();
    let tag_position = opp_pos - ball_to_opp * TAGGING_DISTANCE;

    // Keep some distance from field boundaries
    (
        Some(opponent.id),
        Vector2::new(
            tag_position.x.min(-100.0), // Don't go too far forward
            tag_position.y,
        ),
    )
}

fn find_best_opponent_to_tag(
    s: &RobotSituation,
    passing_store: &PassingStore,
    ball_pos: Vector2,
    current_target_id: Option<PlayerId>,
) -> Option<PlayerData> {
    let mut best_opponent = None;
    let mut best_score = 0.0;
    let mut current_target_score = 0.0;

    // Evaluate all opponents
    for opponent in &s.world.opp_players {
        // Only consider opponents on our side of the field (negative x)
        if opponent.position.x >= 0.0 {
            continue;
        }

        // Skip if opponent is too close to the ball (likely has possession)
        if (opponent.position - ball_pos).norm() < 300.0 {
            continue;
        }

        // Calculate threat score: how dangerous would this opponent be if they had the ball
        let hypothetical_store = passing_store.force_self_position(opponent.position);
        let threat_score = goal_shoot_success_probability(
            &hypothetical_store.swap_team(),
            passing_store.get_own_goal_position(),
        );

        // Prerfer unmarked opponents
        let closest_own_harasser = s
            .get_players_with_role("harasser")
            .into_iter()
            .filter(|p| p.id != s.player_id())
            .min_by_key(|p| (p.position - opponent.position).norm() as i64);
        let unmarked_score = if let Some(closest_own_harasser) = closest_own_harasser {
            if (opponent.position - closest_own_harasser.position).norm() < 400.0 {
                0.3
            } else {
                0.0
            }
        } else {
            0.0
        };
        let final_score = threat_score + unmarked_score;

        // Apply distance penalty (closer opponents are more threatening)
        let distance_to_ball = (opponent.position - s.get_own_goal_position()).norm();
        let distance_factor = 1.0 / (1.0 + distance_to_ball / 1000.0);
        let final_score = final_score * distance_factor;

        if Some(opponent.id) == current_target_id {
            current_target_score = final_score;
        }

        if final_score > best_score {
            best_score = final_score;
            best_opponent = Some(opponent);
        }
    }

    // Apply hysteresis: only switch if new target is significantly better
    if let Some(current_id) = current_target_id {
        if current_target_score > 0.0 && best_score < current_target_score * HYSTERESIS_FACTOR {
            // Keep current target
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
        let ball_pos = ball.position.xy();

        // If ball is in our penalty area, we don't want to pickup the ball
        let margin = 150.0;
        let x_bound = -field.field_length / 2.0 + field.penalty_area_depth - margin;
        let y_bound = field.penalty_area_width / 2.0 - margin;
        if ball_pos.x < x_bound && ball_pos.y.abs() < y_bound {
            return false;
        }

        // Only consider if ball is on our half
        if ball_pos.x >= 70.0 {
            return false;
        }

        // Check if ball is not moving fast
        let ball_velocity = ball.velocity.xy();
        if ball_velocity.norm() > 1000.0 {
            return false;
        }

        s.am_closest_to_ball()
    } else {
        false
    }
}
