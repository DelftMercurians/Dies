use dies_core::Angle;
use dies_executor::behavior_tree_api::{
    BehaviorNode, pick_up_ball, dribble, go_to_position, sequence_node, wait, receive_v2, Argument
};

pub fn build_test_receiverv2() -> BehaviorNode {
    sequence_node()
        .add(go_to_position(Argument::Static(dies_core::Vector2::new(3000.0, 0.0))).with_heading(Argument::Static(Angle::from_degrees(180.0))).build())
        .add(receive_v2(
            Argument::Static(dies_core::Vector2::new(0.0, 0.0)),
            Argument::Static(dies_core::Vector2::new(3000.0, 0.0)),
            Argument::Static(2000.0),
            Argument::Static(true),
        ).build())
        .add(wait(5.0).build())
        .build()
        .into()
}