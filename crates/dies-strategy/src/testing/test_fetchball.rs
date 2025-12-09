use dies_core::Angle;
use dies_executor::behavior_tree_api::{
    BehaviorNode, pick_up_ball, sequence_node, wait, Argument
};

pub fn build_test_fetchball() -> BehaviorNode {
    sequence_node()
        .add(pick_up_ball(Argument::Static(Angle::from_degrees(-135.0))).with_distance_limit(200.0).build())
        .add(wait(3.0).build())
        .build()
        .into()
}
