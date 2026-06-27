use dies_core::Vector2;
use dies_strategy_protocol::GameState;

use crate::bt::*;

use super::utils::get_heading_to_goal;

pub fn build_penalty_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    semaphore_node()
        .do_then(
            sequence_node()
                .add(
                    go_to_position(Argument::callback(|s: &RobotSituation| {
                        s.get_opp_penalty_mark() + Vector2::new(300.0, 0.0)
                    }))
                    .with_heading(Argument::callback(get_heading_to_goal))
                    .description("Approach penalty")
                    .build(),
                )
                .add(
                    guard_node()
                        .condition(|s| {
                            s.game_state_is_one_of(&[GameState::Penalty, GameState::PenaltyRun])
                        })
                        .then(fetch_ball_with_preshoot().with_can_pass(false).build())
                        .description("Can go?")
                        .build(),
                )
                .description("Penalty sequence")
                .build(),
        )
        .semaphore_id("penalty_kicker")
        .max_entry(1)
        .description("Penalty kicker")
        .build()
        .into()
}

pub fn score_penalty_kicker(_s: &RobotSituation) -> f64 {
    90.0
}
