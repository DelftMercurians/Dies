use dies_core::Vector2;
use dies_executor::behavior_tree_api::{
    continuous, select_node, try_receive, BehaviorNode, RobotSituation,
};

pub fn build_test_receiver() -> BehaviorNode {
    select_node()
        .add(try_receive())
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
