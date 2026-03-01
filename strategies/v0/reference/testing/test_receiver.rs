use dies_core::Vector2;
use dies_executor::behavior_tree_api::{
    continuous, fetch_ball_with_preshoot, select_node, sequence_node, try_receive, BehaviorNode,
    RobotSituation,
};

pub fn build_test_receiver() -> BehaviorNode {
    select_node()
        .add(
            sequence_node()
                .add(try_receive())
                .add(fetch_ball_with_preshoot().with_can_pass(false).build())
                .build(),
        )
        .add(
            continuous("position_center")
                .position(position_center)
                .build(),
        )
        .build()
        .into()
}

fn position_center(_s: &RobotSituation) -> Vector2 {
    Vector2::new(1000.0, 500.0)
}
