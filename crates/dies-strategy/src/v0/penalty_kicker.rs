use dies_core::{GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::get_heading_to_goal;

pub fn build_penalty_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    semaphore_node()
        .do_then(
            sequence_node()
                .add(
                    // Move to penalty position
                    go_to_position(Argument::callback(|s| {
                        s.get_opp_penalty_mark() + Vector2::new(-300.0, 0.0)
                    }))
                    .with_heading(Argument::callback(get_heading_to_goal))
                    .description("Approach penalty".to_string())
                    .build(),
                )
                .add(
                    guard_node()
                        .condition(|s| {
                            s.game_state_is_one_of(&[GameState::Penalty, GameState::PenaltyRun])
                        })
                        .then(fetch_ball_with_preshoot().build())
                        .description("Can go?")
                        .build(),
                )
                .description("Penalty sequence")
                .build(),
        )
        .semaphore_id("penalty_kicker".to_string())
        .max_entry(1)
        .description("Penalty kicker")
        .build()
        .into()
}

pub fn score_penalty_kicker(s: &RobotSituation) -> f64 {
    90.0
}
