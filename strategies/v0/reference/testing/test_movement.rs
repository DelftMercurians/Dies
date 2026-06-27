use dies_core::Vector2;
use dies_executor::behavior_tree_api::{
    go_to_position, repeat_node, select_node, semaphore_node, sequence_node, test_movement, wait,
    Argument, BehaviorNode,
};

pub fn build_test_movement() -> BehaviorNode {
    test_movement(
        Argument::Static(Vector2::new(-1200.0, 2000.0)),
        Argument::Static(Vector2::new(-1200.0, -2000.0)),
    )
    .into()
}
