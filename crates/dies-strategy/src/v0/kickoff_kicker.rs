use dies_core::{GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use super::utils::find_best_pass_target;

pub fn build_kickoff_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            guard_node()
                .description("Prepare kickoff?")
                .condition(|s| s.game_state_is(GameState::PrepareKickoff))
                .then(
                    go_to_position(|s: &RobotSituation| {
                        s.get_field_center() - Vector2::new(0.0, 400.0)
                    })
                    .description("Move to center")
                    .build(),
                )
                .build(),
        )
        .add(
            sequence_node()
                .description("Kickoff sequence")
                .add(fetch_ball().description("Get ball".to_string()).build())
                .add(shoot(find_best_shoot_target).build())
                .description("Pass kickoff")
                .build(),
        )
        .build()
        .into()
}
