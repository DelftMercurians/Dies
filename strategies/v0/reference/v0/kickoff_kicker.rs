use dies_core::{GameState, Vector2};
use dies_executor::behavior_tree_api::*;

pub fn build_kickoff_kicker_tree(_s: &RobotSituation) -> BehaviorNode {
    select_node()
        .add(
            guard_node()
                .description("Prepare kickoff?")
                .condition(|s| s.game_state_is(GameState::PrepareKickoff))
                .then(
                    go_to_position(|s: &RobotSituation| {
                        s.get_field_center() - Vector2::new(400.0, 0.0)
                    })
                    .avoid_ball()
                    .description("Move to center")
                    .build(),
                )
                .build(),
        )
        .add(
            fetch_ball_with_preshoot()
                .description("Fetch and shoot".to_string())
                .with_distance_limit(45.0)
                .build(),
        )
        .build()
        .into()
}
