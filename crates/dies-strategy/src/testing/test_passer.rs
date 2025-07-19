use dies_core::PlayerId;
use dies_executor::behavior_tree_api::{
    fetch_ball, fetch_ball_with_preshoot, sequence_node, shoot, to_player, wait, BehaviorNode,
    RobotSituation,
};

pub fn build_test_passer() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball_with_preshoot().build())
        .add(wait(2.0).build())
        .build()
        .into()
}
