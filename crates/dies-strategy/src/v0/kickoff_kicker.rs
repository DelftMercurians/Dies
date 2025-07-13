use dies_executor::behavior_tree_api::*;

use crate::v0::utils::can_pass_to_teammate;

pub fn build_kickoff_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    semaphore_node()
        .do_then(
            sequence_node()
                .add(
                    go_to_position(|s: &RobotSituation| s.get_field_center())
                        .description("Move to center")
                        .build(),
                )
                .add(
                    wait(Argument::Static(0.5))
                        .description("Wait for start".to_string())
                        .build(),
                )
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
                                                    face_position(
                                                        super::utils::find_best_pass_target,
                                                    )
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
                .description("Kickoff sequence")
                .build(),
        )
        .semaphore_id("kickoff_kicker".to_string())
        .max_entry(1)
        .description("Kickoff kicker")
        .build()
        .into()
}
