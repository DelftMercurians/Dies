use dies_executor::behavior_tree_api::{
    at_goal, fetch_ball, sequence_node, shoot, wait, Argument, BehaviorNode,
};

pub fn build_test_fetchball() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball().build())
        .add(shoot(at_goal(Argument::callback(move |s| {
            s.get_own_goal_position()
        }))))
        .add(wait(1.0).build())
        .build()
        .into()
}
