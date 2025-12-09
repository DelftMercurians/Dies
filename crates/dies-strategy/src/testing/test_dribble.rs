use dies_core::Angle;
use dies_executor::behavior_tree_api::{
    BehaviorNode, pick_up_ball, dribble, go_to_position, sequence_node, wait, Argument
};

pub fn build_test_dribble() -> BehaviorNode {
    sequence_node()
        .add(pick_up_ball(Argument::Static(Angle::from_degrees(135.0))).with_distance_limit(200.0).build())
        .add (dribble(
            Argument::Static(dies_core::Vector2::new(   1000.0, 1000.0)),
            Argument::Static(Angle::from_degrees(-90.0)),
            Argument::Static(false),
        ).build())
        .add(go_to_position(Argument::Static(dies_core::Vector2::new(3000.0, 2000.0))).build())
        .add(pick_up_ball(Argument::Static(Angle::from_degrees(-45.0))).with_distance_limit(200.0).build())
        .add (dribble(
            Argument::Static(dies_core::Vector2::new(   -1000.0, -1000.0)),
            Argument::Static(Angle::from_degrees(180.0)),
            Argument::Static(false),
        ).build())
        .add(go_to_position(Argument::Static(dies_core::Vector2::new(2000.0, -2000.0))).build())
        .build()
        .into()

}
