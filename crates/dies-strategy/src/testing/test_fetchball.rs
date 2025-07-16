use dies_executor::behavior_tree_api::{
    fetch_ball_with_preshoot, sequence_node, wait, BehaviorNode,
};

pub fn build_test_fetchball() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball_with_preshoot().build())
        .add(wait(1.0).build())
        .build()
        .into()
}
