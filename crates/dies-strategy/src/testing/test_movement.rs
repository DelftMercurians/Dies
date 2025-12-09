use dies_core::Vector2;
use dies_executor::behavior_tree_api::{
    go_to_position, repeat_node, select_node, semaphore_node, sequence_node, test_movement, wait,
    Argument, BehaviorNode,
};

pub fn build_test_movement() -> BehaviorNode {
    sequence_node()
        .add(go_to_position(Argument::Static(Vector2::new(1000.0, 1000.0))).build())
        .add(wait(3.0).build())
        .add(go_to_position(Argument::Static(Vector2::new(-1000.0, -1000.0))).build())
        .build()
        .into()
}
