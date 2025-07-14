use dies_core::{GameState, Vector2};
use dies_executor::behavior_tree_api::*;

use crate::v0::utils::can_pass_to_teammate;

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
                .add(
                    guard_node()
                        .condition(|s| s.has_ball())
                        .then(
                            select_node()
                                .add(
                                    // Try to pass if teammate available
                                    guard_node()
                                        .condition(can_pass_to_teammate)
                                        .then(
                                            sequence_node()
                                                .add(
                                                    face_position(find_best_pass_target)
                                                        .description("Aim pass".to_string())
                                                        .build(),
                                                )
                                                .add(
                                                    kick()
                                                        .description("Kickoff pass!".to_string())
                                                        .build(),
                                                )
                                                .description("Pass kickoff")
                                                .build(),
                                        )
                                        .description("Can pass")
                                        .build(),
                                )
                                .add(
                                    // Otherwise kick toward goal
                                    sequence_node()
                                        .add(
                                            face_position(|s: &RobotSituation| {
                                                s.get_opp_goal_position()
                                            })
                                            .description("Face goal".to_string())
                                            .build(),
                                        )
                                        .add(kick().description("Kickoff!".to_string()).build())
                                        .description("Direct kickoff")
                                        .build(),
                                )
                                .description("Kickoff options")
                                .build(),
                        )
                        .description("Have ball?")
                        .build(),
                )
                .build(),
        )
        .build()
        .into()
}
