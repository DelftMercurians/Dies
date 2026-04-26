use dies_core::Angle;
use dies_executor::behavior_tree_api::{
    BehaviorNode, pick_up_ball, dribble, go_to_position, sequence_node, wait, kick, Argument
};

pub fn build_test_passerv2() -> BehaviorNode {
    sequence_node()
        .add(pick_up_ball(Argument::Static(Angle::from_degrees(0.0))).with_distance_limit(200.0).build())
        .add(dribble(
            Argument::Static(dies_core::Vector2::new(0.0, 0.0)),
            Argument::Static(Angle::from_degrees(0.0)),
            Argument::Static(false),
        ).build())
        .add(wait(5.0).build())
        .add(kick().build())
        .add(wait(10.0).build())
        .build()
        .into()
}