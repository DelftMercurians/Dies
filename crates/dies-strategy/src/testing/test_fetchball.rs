use dies_core::Vector2;
use dies_executor::behavior_tree_api::{
    at_goal, fetch_ball, go_to_position, repeat_node, select_node, semaphore_node, sequence_node,
    shoot, wait, Argument, BehaviorNode,
};

pub fn build_test_fetchball() -> BehaviorNode {
    sequence_node()
        .add(fetch_ball().build())
        .add(shoot(at_goal(Argument::callback(move |s| {
            s.get_own_goal_position()
        }))))
        .build()
        .into()
}

fn test_movement_seq(index: usize) -> BehaviorNode {
    semaphore_node()
        .semaphore_id(format!("mov_{index}"))
        .max_entry(1)
        .do_then(repeat_node(
            sequence_node()
                .add(
                    go_to_position(Argument::callback(move |s| {
                        Vector2::new(
                            -s.half_field_length() / 2.0 - (index as f64) * 400.0 + 1200.0,
                            2000.0,
                        )
                    }))
                    .build(),
                )
                .add(wait(2.0).build())
                .add(
                    go_to_position(Argument::callback(move |s| {
                        Vector2::new(
                            -s.half_field_length() / 2.0 - (index as f64) * 400.0 + 1200.0,
                            -2000.0,
                        )
                    }))
                    .build(),
                )
                .build(),
        ))
        .build()
        .into()
}
